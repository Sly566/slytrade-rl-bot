"""On-disk partitioned parquet storage for raw bars and ticks.

Layout under ``data_root`` (matches the layout documented in README):

    mt5_bars/symbol=XAUUSDm/timeframe=M1/year=2025/month=01/part-0.parquet
    mt5_ticks/symbol=XAUUSD/year=2025/month=01/day=01.parquet
    exness_ticks/symbol=XAUUSD/year=2025/month=01/part-0.parquet
    merged_ticks/symbol=XAUUSD/year=2025/month=MM/part-0.parquet

Writes use snappy compression, sort by the timestamp key, drop duplicates,
and perform an atomic temp-file rename. All datetime columns are coerced
to tz-aware UTC to keep parquet round-trips stable under pandas 3.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def bar_partition(
    root: Path,
    dataset: str,
    symbol: str,
    timeframe: str,
    year: int,
    month: int,
) -> Path:
    return (
        ensure_dir(root / dataset)
        / f"symbol={symbol}"
        / f"timeframe={timeframe}"
        / f"year={year:04d}"
        / f"month={month:02d}"
    )


def tick_day_partition(root: Path, dataset: str, symbol: str, d: date) -> Path:
    return (
        ensure_dir(root / dataset)
        / f"symbol={symbol}"
        / f"year={d.year:04d}"
        / f"month={d.month:02d}"
    )


def tick_month_partition(root: Path, dataset: str, symbol: str, year: int, month: int) -> Path:
    return (
        ensure_dir(root / dataset)
        / f"symbol={symbol}"
        / f"year={year:04d}"
        / f"month={month:02d}"
    )


# --------------------------------------------------------------------------- #
# Frame hygiene for parquet
# --------------------------------------------------------------------------- #
_DT_COLUMNS = {"time", "time_msc", "decision_time", "quote_time"}


def _normalize_for_parquet(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    """Coerce datetime columns to tz-aware UTC and strip tz-naive garbage."""
    out = df.copy()
    for col in _DT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    # Sort and deduplicate on the primary time column when present.
    if time_col and time_col in out.columns:
        out = out.dropna(subset=[time_col])
        out = out.sort_values(time_col, kind="mergesort")
        out = out.drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)
    return out


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    """Write parquet to a temp file then atomically rename into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".part-", suffix=".parquet", dir=str(target.parent)
    )
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False, compression="snappy")
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
@dataclass
class WriteResult:
    path: Path
    rows: int


def write_partition(
    df: pd.DataFrame,
    target: Path,
    *,
    time_col: str | None = "time",
    filename: str = "part-0.parquet",
) -> WriteResult:
    """Write `df` into a partition directory, merging with any existing file.

    Reads the existing parquet (if present), concatenates, deduplicates and
    sorts by `time_col`, then writes back atomically. Safe to call repeatedly
    with the same partition.
    """
    ensure_dir(target)
    path = target / filename
    if df.empty:
        # Don't overwrite existing data with emptiness.
        rows = 0
        if path.exists():
            try:
                rows = len(pd.read_parquet(path))
            except Exception:
                rows = 0
        return WriteResult(path=path, rows=rows)

    existing = pd.DataFrame()
    if path.exists():
        try:
            existing = pd.read_parquet(path)
        except Exception:
            existing = pd.DataFrame()

    combined = df if existing.empty else pd.concat([existing, df], ignore_index=True)
    combined = _normalize_for_parquet(combined, time_col=time_col)
    _atomic_write_parquet(combined, path)
    return WriteResult(path=path, rows=len(combined))


def write_day_partition(
    df: pd.DataFrame,
    target: Path,
    d: date,
) -> WriteResult:
    """Per-day tick parquet (one file per calendar day)."""
    return write_partition(
        df, target, time_col="time_msc", filename=f"{d.day:02d}.parquet"
    )


def read_partitions(
    root: Path,
    pattern: str = "**/*.parquet",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read every parquet file under `root` matching `pattern` into one frame."""
    if not root.exists():
        return pd.DataFrame()
    files = sorted(root.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, columns=columns))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in _DT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def discover_partitions(root: Path, pattern: str = "**/*.parquet") -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob(pattern))
