"""MT5-side collectors for bars (monthly) and ticks (daily with early-stop).

Design rules baked in:
* `mt5.initialize()` is already handled by the MT5Client before each call;
  these collectors simply invoke `client.get_bars` / `client.get_ticks` which
  themselves call initialize().
* Bars are fetched one calendar month per chunk (MT5 returns ~30-31 days of
  M1 cleanly) and written into the Hive-style partition layout.
* Ticks are fetched one calendar day per chunk. Brokers typically only keep
  ~6-8 months of tick history on demo; we probe BACKWARD from end until we
  see `empty_streak_stop` consecutive empty days, then only fetch forward from
  that cliff. The probe cache avoids re-fetching the same chunks twice.
* All paths use the ``DataConfig`` properties so the caller controls layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from ..brokers.mt5_adapter import MT5Client
from ..config import DataConfig
from .storage import (
    bar_partition,
    tick_day_partition,
    write_day_partition,
    write_partition,
)
from .time import iter_days, iter_months


ProgressFn = Callable[[str], None]


# --------------------------------------------------------------------------- #
# Bar collector
# --------------------------------------------------------------------------- #
@dataclass
class BarCollectionStats:
    per_tf_rows: dict[str, int]
    per_tf_files: dict[str, int]


class MT5BarCollector:
    """Collect OHLCV bars for every requested timeframe, one month per file."""

    def __init__(self, client: MT5Client, data_cfg: DataConfig, progress: ProgressFn | None = None):
        self.client = client
        self.data = data_cfg
        self.progress = progress or (lambda _msg: None)

    def collect(
        self,
        *,
        symbol: str,
        timeframes: list[str],
        lookback_years: float,
        raw_symbol: str,
        clean: bool = False,
    ) -> dict[str, int]:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=int(lookback_years * 365))

        if clean:
            bars_root = self.data.mt5_bars_path / f"symbol={raw_symbol}"
            if bars_root.exists():
                import shutil

                shutil.rmtree(bars_root, ignore_errors=True)

        counts: dict[str, int] = {tf: 0 for tf in timeframes}

        for tf in timeframes:
            tf_rows = 0
            file_count = 0
            for m_start, _m_end in iter_months(start_dt.date(), end_dt.date()):
                # Extend chunk to min(next_month, now) so the current month is included.
                y, mo = m_start.year, m_start.month
                if mo == 12:
                    chunk_end = date(y + 1, 1, 1)
                else:
                    chunk_end = date(y, mo + 1, 1)
                chunk_start_dt = datetime(y, mo, 1, tzinfo=timezone.utc)
                chunk_end_dt = datetime.combine(chunk_end, datetime.min.time(), tzinfo=timezone.utc)
                chunk_end_dt = min(chunk_end_dt, end_dt)

                try:
                    df = self.client.get_bars(raw_symbol, tf, chunk_start_dt, chunk_end_dt)
                except Exception as e:
                    self.progress(f"    {tf} {y}-{mo:02d} fetch error: {e}")
                    continue

                if df is None or df.empty:
                    continue

                part_dir = bar_partition(
                    self.data.mt5_bars_path, "", raw_symbol, tf, y, mo
                )
                # write_partition appends to any existing parquet in that dir.
                res = write_partition(df, part_dir, time_col="time")
                tf_rows += len(df)
                file_count += 1

                if file_count % 50 == 0:
                    self.progress(f"    {tf}: {tf_rows:,} rows / {file_count} files ...")

            counts[tf] = tf_rows
            self.progress(f"    {tf}: {tf_rows:,} rows / {file_count} files written")

        return counts


# --------------------------------------------------------------------------- #
# Tick collector — with backward-probe early stop
# --------------------------------------------------------------------------- #
@dataclass
class TickCollectionResult:
    total_rows: int
    start_date: date | None  # earliest day with data found


class MT5TickCollector:
    """Collect ticks one day per file, with an MT5-history-cliff early stop."""

    def __init__(
        self,
        client: MT5Client,
        data_cfg: DataConfig,
        *,
        empty_streak_stop: int = 30,
        progress: ProgressFn | None = None,
        progress_every_files: int = 20,
        progress_every_chunks: int = 30,
    ):
        self.client = client
        self.data = data_cfg
        self.empty_streak_stop = empty_streak_stop
        self.progress = progress or (lambda _msg: None)
        self.progress_every_files = progress_every_files
        self.progress_every_chunks = progress_every_chunks

    # ------------------------------------------------------------------ #
    def _fetch_day(self, raw_symbol: str, d: date) -> pd.DataFrame:
        start_dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=1) - timedelta(microseconds=1)
        return self.client.get_ticks(raw_symbol, start_dt, end_dt)

    # ------------------------------------------------------------------ #
    def collect(
        self,
        *,
        symbol: str,
        lookback_years: float,
        raw_symbol: str,
        clean: bool = False,
    ) -> tuple[int, date | None]:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=int(lookback_years * 365))
        start_date = start_dt.date()
        end_date = end_dt.date()

        ticks_root = self.data.mt5_ticks_path / f"symbol={raw_symbol}"
        if clean and ticks_root.exists():
            import shutil

            shutil.rmtree(ticks_root, ignore_errors=True)

        all_days: list[date] = list(iter_days(start_date, end_date))
        if not all_days:
            return 0, None

        # ---- Phase 1: probe BACKWARD from end until N consecutive empties ---- #
        probe_cache: dict[date, pd.DataFrame] = {}
        consec_empty = 0
        found_any = False
        effective_start_idx = len(all_days)  # default: all skipped
        cliff: date | None = None

        if self.empty_streak_stop > 0 and len(all_days) > self.empty_streak_stop:
            self.progress("    probing backward to locate MT5 history cliff ...")
            for idx in range(len(all_days) - 1, -1, -1):
                d = all_days[idx]
                try:
                    df = self._fetch_day(raw_symbol, d)
                except Exception:
                    df = pd.DataFrame()
                probe_cache[d] = df
                if df is None or df.empty:
                    consec_empty += 1
                    if found_any and consec_empty >= self.empty_streak_stop:
                        # first real data is at idx+1
                        effective_start_idx = idx + 1
                        cliff = all_days[effective_start_idx] if effective_start_idx < len(all_days) else None
                        break
                else:
                    consec_empty = 0
                    found_any = True
            if cliff is not None:
                skipped = effective_start_idx
                self.progress(
                    f"    broker tick history starts {cliff.isoformat()}; "
                    f"skipping {skipped} empty day-chunk(s) before that."
                )
                # Drop cached empties before the cliff to free memory.
                for pre_idx in range(effective_start_idx):
                    probe_cache.pop(all_days[pre_idx], None)

        # ---- Phase 2: fetch forward from effective_start_idx ---- #
        total_rows = 0
        files_written = 0
        chunks_attempted = 0
        first_data_day: date | None = None

        for idx in range(effective_start_idx, len(all_days)):
            d = all_days[idx]
            chunks_attempted += 1

            if d in probe_cache:
                df = probe_cache.pop(d)
            else:
                try:
                    df = self._fetch_day(raw_symbol, d)
                except Exception as e:
                    if chunks_attempted % self.progress_every_chunks == 0:
                        self.progress(f"    {d.isoformat()} fetch error: {e}")
                    continue

            if df is None or df.empty:
                continue

            part_dir = tick_day_partition(self.data.mt5_ticks_path, "", raw_symbol, d)
            try:
                write_day_partition(df, part_dir, d)
            except Exception as e:
                self.progress(f"    {d.isoformat()} write error: {e}")
                continue

            n = len(df)
            total_rows += n
            files_written += 1
            if first_data_day is None:
                first_data_day = d

            if files_written % self.progress_every_files == 0 or chunks_attempted % self.progress_every_chunks == 0:
                self.progress(
                    f"    {total_rows:,} rows / {files_written} files so far "
                    f"(through {d.isoformat()})..."
                )

        return total_rows, first_data_day
