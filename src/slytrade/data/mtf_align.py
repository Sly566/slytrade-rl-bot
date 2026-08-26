"""Layer 3 — Multi-Timeframe causal alignment.

Takes per-TF processed bars (from ``data/processed/bars/``) and produces a
single M1-indexed frame where every higher-timeframe (M5, M15, M30, H1,
H4, D1, W1) feature column is asof-joined with ``direction='backward'``
so that an M1 bar at time ``t`` only ever sees HTF information from HTF
bars that had **already closed** by ``t``.

Causal rule
-----------
Let ``htf_dur`` = duration of one HTF bar.  An HTF bar with open time
``h_open`` closes at ``h_open + htf_dur`` = ``decision_time``.  When we
join an M1 bar whose open time is ``m1_open``, we select the latest HTF
bar whose ``decision_time <= m1_open``.  The HTF bar that closes exactly
at an M1 open (e.g. M5 bar 10:00-10:05 closing at 10:05, M1 bar 10:05
opening) is visible to that M1 bar — the HTF bar is fully formed before
any in-bar action on the M1 occurs.

Output partition layout
-----------------------
::

    data/processed/aligned/symbol=XAUUSDm/year=YYYY/month=MM/part-0.parquet

Same monthly partition scheme as processed M1 bars (the aligned frame is
M1-indexed).

HTF columns prefixed and carried across
---------------------------------------
All structural/volatility/EMA/OB/FVG/premium-discount/displacement/
candle-pattern features from Layer 2 are carried across, plus
``{TF}_close``.  Dropped before prefixing: raw OHLC (except close),
volume/spread/real_volume, session/temporal columns (identical on M1),
and intra-TF positional ``*_idx`` columns.

A debug column ``{TF}_bar_time`` is added = the OPEN time of the HTF bar
whose features are attached, for causality sanity-checks.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq

from ..config import DataConfig
from .storage import (
    _atomic_write_parquet,
    _normalize_for_parquet,
    bar_partition,
    discover_partitions,
    ensure_dir,
)
from .time import _common_dt_unit, timeframe_timedelta

ProgressFn = Callable[[str], None]

EXECUTION_TF: str = "M1"

# Higher timeframes to align onto M1, from fastest to slowest.
HTFS: list[str] = ["M5", "M15", "M30", "H1", "H4", "D1", "W1"]

# Columns to DROP from each HTF frame before prefixing.
_HTF_DROP_COLS: frozenset[str] = frozenset({
    "open", "high", "low",
    "tick_volume", "spread", "real_volume",
    "session", "hour", "minute", "dow",
    "kz_asian", "kz_london", "kz_ny",
    "london_open_30", "ny_open_30",
    "minor_swing_high_idx", "minor_swing_low_idx",
    "major_swing_high_idx", "major_swing_low_idx",
    "bull_ob_idx", "bear_ob_idx",
    "bull_fvg_idx", "bear_fvg_idx",
})


@dataclass
class AlignResult:
    rows: int
    files: int
    columns: int
    per_tf_cols: dict[str, int]


def _list_processed_partitions(root: Path, symbol: str, tf: str) -> list[tuple[int, int, Path]]:
    """Return [(year, month, path)] of existing processed-bar partitions."""
    base = root / f"symbol={symbol}" / f"timeframe={tf}"
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


def _read_processed_tf(paths: list[Path], tf: str) -> pd.DataFrame:
    """Read+concat all processed parquet partitions for one TF.

    Adds ``decision_time`` = time + bar_duration (the close time, when
    this bar's information becomes available to lower TFs).
    """
    frames: list[pd.DataFrame] = []
    for p in sorted(paths):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)
    df["decision_time"] = df["time"] + timeframe_timedelta(tf)
    return df


def _prep_htf_frame(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Prepare an HTF frame for asof-joining:

    1. Drop columns we don't carry across (OHLC except close, volume,
       session/temporal, *_idx).
    2. Keep ``decision_time`` as the join key.
    3. Rename ``time`` to ``{tf}_bar_time`` for debugging.
    4. Prefix every remaining column with ``{tf}_``.
    """
    if df.empty:
        return pd.DataFrame(columns=["decision_time"])

    keep_cols: list[str] = []
    for c in df.columns:
        if c in ("decision_time", "time"):
            continue
        if c in _HTF_DROP_COLS:
            continue
        keep_cols.append(c)

    out = pd.DataFrame()
    out["decision_time"] = df["decision_time"]
    out[f"{tf}_bar_time"] = df["time"]
    for c in keep_cols:
        out[f"{tf}_{c}"] = df[c]

    out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out[f"{tf}_bar_time"] = pd.to_datetime(out[f"{tf}_bar_time"], utc=True, errors="coerce")
    return out


def _asof_merge(left: pd.DataFrame, right: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Merge `right` (HTF) onto `left` with strict causal backward-asof:

        left.time  <-backward-  right.decision_time

    Uses Pandas-3-safe ns casting on both sides.  The ``decision_time``
    column from the right frame is NOT carried into the output (it was
    only ever the join key; ``{tf}_bar_time`` is the user-facing debug
    column).  This prevents suffix collisions when chaining merges.
    """
    if right.empty or "decision_time" not in right.columns:
        return left

    left_key, right_key = _common_dt_unit(left["time"], right["decision_time"])
    left = left.copy()
    left["_join_key"] = left_key
    right = right.copy()
    right["_join_key"] = right_key
    # Drop decision_time from right BEFORE merge so it doesn't appear in
    # output and collide on subsequent merges.
    right = right.drop(columns=["decision_time"])

    merged = pd.merge_asof(
        left.sort_values("_join_key", kind="mergesort"),
        right.sort_values("_join_key", kind="mergesort"),
        on="_join_key",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.drop(columns=["_join_key"])
    return merged


def align_all(
    data_cfg: DataConfig,
    symbol: str,
    raw_symbol: str,
    *,
    clean: bool = False,
    progress: ProgressFn | None = None,
) -> AlignResult:
    """Align all HTF processed bars onto M1, writing monthly parquet to
    ``data/processed/aligned/``.

    Memory strategy: load every HTF into memory once (combined HTF data
    is ~100-150 MB for a 2y XAUUSD dataset — H1+ have tiny row counts),
    then stream M1 month-by-month through ``merge_asof``.  Peak memory
    is one M1 month × (~500 cols) ≈ 100-150 MB.
    """
    progress = progress or (lambda _m: None)

    processed_bars = data_cfg.processed_bars_path
    aligned_root = data_cfg.aligned_path
    if clean and aligned_root.exists():
        shutil.rmtree(aligned_root, ignore_errors=True)
    ensure_dir(aligned_root)

    # ---- Load every HTF into memory ----
    progress("Loading HTF processed bars into memory ...")
    htf_frames: dict[str, pd.DataFrame] = {}
    per_tf_cols: dict[str, int] = {}
    for tf in HTFS:
        parts = _list_processed_partitions(processed_bars, raw_symbol, tf)
        if not parts:
            progress(f"  {tf}: no processed partitions found - skipping.")
            htf_frames[tf] = pd.DataFrame(columns=["decision_time"])
            per_tf_cols[tf] = 0
            continue
        df = _read_processed_tf([p for (_y, _m, p) in parts], tf)
        prepped = _prep_htf_frame(df, tf)
        htf_frames[tf] = prepped
        # Subtract 1 for the join key (decision_time) which we drop during merge.
        col_count = len(prepped.columns) - 1
        per_tf_cols[tf] = col_count
        progress(f"  {tf}: {len(df):,} bars loaded, {col_count} feature columns to join.")

    # ---- M1 partitions ----
    m1_parts = _list_processed_partitions(processed_bars, raw_symbol, EXECUTION_TF)
    if not m1_parts:
        progress(f"  {EXECUTION_TF}: no processed partitions found - nothing to align.")
        return AlignResult(rows=0, files=0, columns=0, per_tf_cols=per_tf_cols)
    progress(f"Processing {len(m1_parts)} M1 month-partition(s) ...")

    total_rows = 0
    total_files = 0
    out_col_count = 0

    for (y, m, part_path) in m1_parts:
        m1_df = pd.read_parquet(part_path)
        m1_df["time"] = pd.to_datetime(m1_df["time"], utc=True, errors="coerce")
        m1_df = m1_df.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)
        if m1_df.empty:
            progress(f"    M1 {y}-{m:02d}: empty after loading - skipping.")
            continue

        aligned = m1_df
        for tf in HTFS:
            aligned = _asof_merge(aligned, htf_frames[tf], tf)

        # Belt-and-suspenders: drop any stray decision_time columns.
        drop_cols = [c for c in aligned.columns if c == "decision_time" or c.startswith("decision_time_")]
        if drop_cols:
            aligned = aligned.drop(columns=drop_cols)

        aligned = aligned.sort_values("time", kind="mergesort").reset_index(drop=True)

        part_dir = bar_partition(aligned_root, "", raw_symbol, EXECUTION_TF, y, m)
        target = part_dir / "part-0.parquet"
        out_df = _normalize_for_parquet(aligned, time_col="time")
        _atomic_write_parquet(out_df, target)

        n = len(out_df)
        total_rows += n
        total_files += 1
        out_col_count = len(out_df.columns)
        progress(f"    M1 {y}-{m:02d}: {n:,} rows x {out_col_count} cols written.")

    return AlignResult(
        rows=total_rows,
        files=total_files,
        columns=out_col_count,
        per_tf_cols=per_tf_cols,
    )


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@dataclass
class AlignedDiagnostics:
    files: int = 0
    rows: int = 0
    columns: int = 0
    start: str = ""
    end: str = ""
    htf_cols: dict[str, int] | None = None
    issues: list[str] | None = None

    def __post_init__(self) -> None:
        if self.htf_cols is None:
            self.htf_cols = {}
        if self.issues is None:
            self.issues = []


def inspect_aligned(data_cfg: DataConfig, raw_symbol: str) -> AlignedDiagnostics:
    """Inspect aligned parquet partitions. Streams to avoid OOM."""
    diag = AlignedDiagnostics()
    align_root = data_cfg.aligned_path
    base = align_root / f"symbol={raw_symbol}" / f"timeframe={EXECUTION_TF}"
    if not base.exists():
        diag.issues.append("no aligned data found (run `slytrade align`)")
        return diag

    files = discover_partitions(base, "**/*.parquet")
    diag.files = len(files)
    if not files:
        diag.issues.append("no aligned parquet files")
        return diag

    total_rows = 0
    col_names: list[str] = []
    first_start: pd.Timestamp | None = None
    last_end: pd.Timestamp | None = None
    prev_end: pd.Timestamp | None = None
    monotonic = True

    htf_counts: dict[str, int] = {tf: 0 for tf in HTFS}
    causality_violations = 0

    for f in sorted(files):
        try:
            pf = pq.ParquetFile(str(f))
        except Exception as e:
            diag.issues.append(f"read error: {f.name}: {e}")
            continue

        if not col_names:
            col_names = pf.schema.names
            diag.columns = len(col_names)
            for tf in HTFS:
                prefix = f"{tf}_"
                htf_counts[tf] = sum(1 for c in col_names if c.startswith(prefix))
            diag.htf_cols = htf_counts

        # Stream time + HTF bar_time columns for causality check.
        time_cols = ["time"] + [f"{tf}_bar_time" for tf in HTFS if f"{tf}_bar_time" in col_names]
        for batch in pf.iter_batches(batch_size=2_000_000, columns=time_cols):
            chunk = batch.to_pandas()
            chunk["time"] = pd.to_datetime(chunk["time"], utc=True, errors="coerce")
            chunk = chunk.dropna(subset=["time"])
            if chunk.empty:
                continue
            n = len(chunk)
            total_rows += n
            t_min = chunk["time"].min()
            t_max = chunk["time"].max()
            if first_start is None or t_min < first_start:
                first_start = t_min
            if last_end is None or t_max > last_end:
                last_end = t_max
            if prev_end is not None and t_min < prev_end:
                monotonic = False
            prev_end = t_max

            # Causality: for every HTF, bar_time + bar_duration must be <= time.
            # Since we joined on decision_time (= bar_time + dur) <= time with
            # direction='backward', any violation means a bug.  NaN bar_time
            # means "no HTF bar yet at start of data" which is fine.
            for tf in HTFS:
                bt_col = f"{tf}_bar_time"
                if bt_col not in chunk.columns:
                    continue
                chunk[bt_col] = pd.to_datetime(chunk[bt_col], utc=True, errors="coerce")
                dur = timeframe_timedelta(tf)
                # Allow 1 ms tolerance for timestamp rounding in parquet.
                strict_bad = (
                    chunk[bt_col].notna()
                    & ((chunk[bt_col] + dur) > chunk["time"] + pd.Timedelta(milliseconds=1))
                )
                causality_violations += int(strict_bad.sum())

    diag.rows = total_rows
    diag.start = str(first_start)[:19] if first_start is not None else "-"
    diag.end = str(last_end)[:19] if last_end is not None else "-"
    if not monotonic:
        diag.issues.append("timestamps not monotonic across partitions")
    if total_rows == 0:
        diag.issues.append("zero rows")
    if causality_violations > 0:
        diag.issues.append(f"{causality_violations} causality violations (HTF bar not closed before M1 bar)")
    return diag
