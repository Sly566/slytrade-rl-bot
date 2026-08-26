"""Per-timeframe feature processing driver.

Reads raw MT5 bars for each timeframe (from data/raw/mt5_bars), runs
``features.process_bars`` to compute strictly-causal ICT/SMC features, and
writes the result to ``data/processed/bars/`` using the same Hive-style
partition layout as the raw data.

Partition layout::

    processed/bars/symbol=XAUUSDm/timeframe=M1/year=2025/month=01/part-0.parquet

The processor is idempotent: re-running overwrites each processed partition
with freshly-computed features from the corresponding raw partition, so
when you collect new bars you just re-process.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..config import DataConfig
from .features import DEFAULT_CONFIG, FeatureConfig, process_bars
from .storage import _atomic_write_parquet, _normalize_for_parquet, bar_partition, ensure_dir

ProgressFn = Callable[[str], None]


@dataclass
class ProcessResult:
    per_tf_rows: dict[str, int]
    per_tf_files: dict[str, int]


def _list_raw_bar_partitions(raw_root: Path, symbol: str, tf: str) -> list[tuple[int, int, Path]]:
    """Return [(year, month, path)] of existing raw bar partitions for a TF."""
    base = raw_root / "mt5_bars" / f"symbol={symbol}" / f"timeframe={tf}"
    out: list[tuple[int, int, Path]] = []
    if not base.exists():
        return out
    for y_dir in sorted(base.glob("year=*")):
        try:
            y = int(y_dir.name.split("=", 1)[1])
        except ValueError:
            continue
        for m_dir in sorted(y_dir.glob("month=*")):
            try:
                m = int(m_dir.name.split("=", 1)[1])
            except ValueError:
                continue
            part_file = m_dir / "part-0.parquet"
            if part_file.exists() and part_file.stat().st_size > 0:
                out.append((y, m, part_file))
    return out


def _read_partition(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Pandas-3 safe: ensure time is tz-aware UTC.
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)
    return df


def process_all(
    data_cfg: DataConfig,
    symbol: str,
    raw_symbol: str,
    timeframes: list[str],
    *,
    feature_cfg: FeatureConfig | None = None,
    clean: bool = False,
    progress: ProgressFn | None = None,
) -> ProcessResult:
    """Process every raw-bar month-partition for each timeframe.

    Reads raw bars per-month (so memory stays bounded by a month of M1 ~30-45k
    rows × 87 cols ≈ 20 MB), computes features, writes per-month parquet to
    the processed tree.
    """
    progress = progress or (lambda _m: None)
    feature_cfg = feature_cfg or DEFAULT_CONFIG

    out_root = data_cfg.processed_bars_path
    if clean and out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    ensure_dir(out_root)

    results = ProcessResult(per_tf_rows={}, per_tf_files={})

    for tf in timeframes:
        tf_rows = 0
        tf_files = 0
        parts = _list_raw_bar_partitions(data_cfg.raw_root, raw_symbol, tf)
        if not parts:
            progress(f"  {tf}: no raw partitions found, skipping")
            results.per_tf_rows[tf] = 0
            results.per_tf_files[tf] = 0
            continue
        progress(f"  {tf}: {len(parts)} month-partition(s) to process")
        # We MUST process months in chronological order because indicators
        # like EMA/ATR/wilders need a warm-up from earlier data. To keep
        # memory bounded per month we carry over the tail of the previous
        # month's processed data (enough for the longest lookback, plus a
        # safety buffer) to seed indicator state via the indicator's own
        # ewm/rolling windows, then trim the overlap away before write.
        #
        # However, ewm/rolling don't accept "seed state" directly — so we
        # simply concatenate a warm-up slice of prior processed data onto
        # the raw month, run process_bars on the combined frame, and write
        # out only the rows belonging to the current month. This gives
        # continuous indicator values across month boundaries.
        warmup_len = max(feature_cfg.ema_slow, feature_cfg.atr_smooth_len) + 10
        prior_tail = pd.DataFrame()
        for (y, m, part_path) in parts:
            raw_df = _read_partition(part_path)
            if raw_df.empty:
                continue
            if not prior_tail.empty:
                combined = pd.concat([prior_tail, raw_df], ignore_index=True)
            else:
                combined = raw_df
            processed = process_bars(
                combined, tf, feature_cfg,
                progress=None,  # silence per-part noise
            )
            # Keep only rows whose time falls within the target month.
            month_start = datetime(y, m, 1, tzinfo=UTC)
            if m == 12:
                next_month_start = datetime(y + 1, 1, 1, tzinfo=UTC)
            else:
                next_month_start = datetime(y, m + 1, 1, tzinfo=UTC)
            mask = (processed["time"] >= month_start) & (processed["time"] < next_month_start)
            out_df = processed.loc[mask].reset_index(drop=True)
            if out_df.empty:
                progress(f"    {tf} {y}-{m:02d}: no rows in window after processing")
                continue
            # Write partition.
            part_dir = bar_partition(out_root, "", raw_symbol, tf, y, m)
            target = part_dir / "part-0.parquet"
            out_df = _normalize_for_parquet(out_df, time_col="time")
            _atomic_write_parquet(out_df, target)
            n = len(out_df)
            tf_rows += n
            tf_files += 1
            # Save tail for next month's warm-up (use last warmup_len rows
            # of the processed frame, which includes this month's close).
            prior_tail = processed.tail(warmup_len).reset_index(drop=True)
            progress(f"    {tf} {y}-{m:02d}: {n:,} rows written")
        results.per_tf_rows[tf] = tf_rows
        results.per_tf_files[tf] = tf_files
        progress(f"  {tf}: {tf_rows:,} rows / {tf_files} files total")
    return results
