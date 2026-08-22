"""Streaming MT5 + Exness tick merger (authoritative MT5 wins on overlap).

Merge strategy (matches the spec from the handoff doc):

    merged = Exness-only months  ∪  (Exness before MT5-cliff  ⊕  MT5 from cliff)

* MT5 ticks are authoritative for any period they cover.
* Exness ticks fill history before MT5's earliest available day.
* Each calendar month is processed independently so memory is O(one month).
* Output layout: ``data/raw/merged_ticks/symbol=SYM/year=YYYY/month=MM/part-0.parquet``
  with columns: time_msc, bid, ask, last, volume, flags, spread, mid.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import DataConfig
from .exness_archive import normalize_symbol
from .storage import (
    discover_partitions,
    read_partitions,
    tick_month_partition,
    write_partition,
)
from .time import ensure_utc, iter_months


ProgressFn = Callable[[str], None]


# --------------------------------------------------------------------------- #
# File discovery / month keying
# --------------------------------------------------------------------------- #
def _parse_int(part: str, key: str) -> int:
    raw = part.split("=", 1)[1].split(".", 1)[0]
    return int(raw)


def _month_key_from_path(p: Path) -> tuple[int, int]:
    year = month = 0
    for seg in p.parts:
        if seg.startswith("year="):
            year = _parse_int(seg, "year")
        elif seg.startswith("month="):
            month = _parse_int(seg, "month")
    return (year, month)


def _to_ns(s: pd.Series) -> pd.Series:
    """Convert a datetime Series to ns resolution.

    Pandas 3 defaults to ``us``; ``merge_asof`` and min/max comparisons
    across files produced in different pandas versions need a common unit.
    Using the DatetimeArray's ``as_unit('ns')`` avoids the 1000× tick-rate bug
    that came from casting the raw int view.
    """
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return pd.DatetimeIndex(dt.array.as_unit("ns"))


def _min_tick_time(files: list[Path]) -> pd.Timestamp | None:
    """Cheap minimum time across parquet files (only reads time_msc)."""
    best: pd.Timestamp | None = None
    for f in files:
        try:
            col = pd.read_parquet(f, columns=["time_msc"])["time_msc"]
        except Exception:
            continue
        if col.empty:
            continue
        col = _to_ns(col)
        m = col.min()
        if pd.isna(m):
            continue
        if best is None or m < best:
            best = m
    return best


def _read_tick_file(f: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(f)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["time_msc"] = pd.to_datetime(df["time_msc"], utc=True, errors="coerce")
    return df.dropna(subset=["time_msc"])


# --------------------------------------------------------------------------- #
# TickMerger
# --------------------------------------------------------------------------- #
@dataclass
class MergeResult:
    total_rows: int = 0
    months: int = 0
    months_from_mt5_only: int = 0
    months_from_exness_only: int = 0
    months_merged: int = 0
    months_empty: int = 0


class TickMerger:
    """Merge MT5 and Exness ticks month-by-month into merged_ticks/."""

    def __init__(self, data_cfg: DataConfig, *, progress: ProgressFn | None = None):
        self.data = data_cfg
        self.progress = progress or (lambda _m: None)

    # ------------------------------------------------------------------ #
    def _files_by_month(self, dataset_dir: Path, symbol: str) -> dict[tuple[int, int], list[Path]]:
        root = dataset_dir / f"symbol={symbol}"
        out: dict[tuple[int, int], list[Path]] = defaultdict(list)
        for f in discover_partitions(root, "**/*.parquet"):
            out[_month_key_from_path(f)].append(f)
        return out

    # ------------------------------------------------------------------ #
    def merge_range(
        self,
        *,
        symbol: str,
        start: date | datetime,
        end: date | datetime,
        clean: bool = False,
    ) -> MergeResult:
        if isinstance(start, datetime):
            start = ensure_utc(start).date()
        if isinstance(end, datetime):
            end = ensure_utc(end).date()

        sym = normalize_symbol(symbol)
        out_root = self.data.merged_ticks_path / f"symbol={sym}"

        if clean and out_root.exists():
            import shutil

            shutil.rmtree(out_root, ignore_errors=True)

        mt5_by_month = self._files_by_month(self.data.mt5_ticks_path, sym)
        # MT5 ticks are stored per-day under the same root; check the raw
        # (possibly broker-suffixed) symbol dirs if the normalised symbol has
        # no files.
        if not mt5_by_month:
            for candidate in (sym + "m", sym + ".m", sym):
                mt5_by_month = self._files_by_month(self.data.mt5_ticks_path, candidate)
                if mt5_by_month:
                    break

        exness_by_month = self._files_by_month(self.data.exness_ticks_path, sym)

        # Find the earliest MT5 tick — Exness is only used before that cutoff.
        mt5_files = [f for flist in mt5_by_month.values() for f in flist]
        cutoff = _min_tick_time(mt5_files) if mt5_files else None
        if cutoff is not None:
            self.progress(
                f"  MT5 tick coverage starts {cutoff.date().isoformat()}; "
                f"Exness will fill before that."
            )

        result = MergeResult()
        months = list(iter_months(start, end))
        for m_start, _m_end in months:
            y, mo = m_start.year, m_start.month
            key = (y, mo)
            frames: list[pd.DataFrame] = []

            # Exness before cutoff only
            for f in exness_by_month.get(key, []):
                df = _read_tick_file(f)
                if df.empty:
                    continue
                if cutoff is not None:
                    df = df[df["time_msc"] < cutoff]
                if not df.empty:
                    frames.append(df)

            # MT5 always
            for f in mt5_by_month.get(key, []):
                df = _read_tick_file(f)
                if not df.empty:
                    frames.append(df)

            result.months += 1

            if not frames:
                result.months_empty += 1
                continue

            has_exness = any(
                (cutoff is None or True) and not _read_tick_file(f).empty
                for f in exness_by_month.get(key, [])
            ) if exness_by_month.get(key) else False
            has_mt5 = bool(mt5_by_month.get(key))

            if has_exness and has_mt5 and cutoff is not None:
                result.months_merged += 1
            elif has_mt5:
                result.months_from_mt5_only += 1
            elif has_exness:
                result.months_from_exness_only += 1

            month_df = pd.concat(frames, ignore_index=True)
            month_df["time_msc"] = pd.to_datetime(month_df["time_msc"], utc=True, errors="coerce")
            month_df = (
                month_df.dropna(subset=["time_msc"])
                .sort_values("time_msc", kind="mergesort")
                .drop_duplicates(subset=["time_msc"], keep="last")
                .reset_index(drop=True)
            )
            # Make sure all canonical columns exist.
            for col, default in [
                ("last", 0.0),
                ("volume", 0.0),
                ("flags", 0),
                ("spread", 0.0),
                ("mid", 0.0),
            ]:
                if col not in month_df.columns:
                    month_df[col] = default
            # Recompute spread/mid from bid/ask where they're zero/missing.
            month_df["spread"] = month_df["ask"] - month_df["bid"]
            month_df["mid"] = (month_df["bid"] + month_df["ask"]) / 2.0
            month_df["symbol"] = sym

            keep = ["time_msc", "bid", "ask", "last", "volume", "flags", "spread", "mid", "symbol"]
            month_df = month_df[keep]

            part_dir = tick_month_partition(self.data.merged_ticks_path, "", sym, y, mo)
            write_partition(month_df, part_dir, time_col="time_msc")
            result.total_rows += len(month_df)

            if result.months % 5 == 0:
                self.progress(
                    f"  merged {result.total_rows:,} rows across "
                    f"{result.months} month(s) ..."
                )

        return result
