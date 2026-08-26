"""Streaming signal scanner for aligned M1 parquets.

Walks monthly aligned partitions one-at-a-time, selecting only the columns
needed by the strategy. This keeps peak memory bounded by one M1 month (~30k
rows × ~200 cols ≈ 50 MB) instead of loading the full 2y × 535-col frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import DataConfig
from ..data.storage import discover_partitions, _normalize_for_parquet, _atomic_write_parquet
from .config import StrategyConfig
from .signals import Signal, _strategy_columns, _evaluate_row


ProgressFn = Callable[[str], None]


@dataclass
class ScanResult:
    signals: List[Signal]
    rows_scanned: int


# --------------------------------------------------------------------------- #
# Vectorized numpy-driven scanner
# --------------------------------------------------------------------------- #
#
# Instead of iterating row-by-row in Python, we build numpy arrays for the
# columns we need and do fast vectorized pre-filtering, only calling into the
# richer per-row logic when the trigger/session conditions are met.
#

def _col(df: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=np.float64, copy=False)
    return np.full(len(df), default, dtype=np.float64)


def _bcol(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].fillna(False).to_numpy(dtype=bool, copy=False)
    return np.zeros(len(df), dtype=bool)


def _icol(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].fillna(0).to_numpy(dtype=np.int8, copy=False)
    return np.zeros(len(df), dtype=np.int8)


def scan_aligned(
    data_cfg: DataConfig,
    raw_symbol: str,
    cfg: Optional[StrategyConfig] = None,
    progress: Optional[ProgressFn] = None,
) -> ScanResult:
    """Stream-scan every aligned M1 partition and return signals."""
    cfg = cfg or StrategyConfig()
    progress = progress or (lambda _m: None)

    base = data_cfg.aligned_path / f"symbol={raw_symbol}" / "timeframe=M1"
    files = discover_partitions(base, "**/*.parquet")
    if not files:
        progress("No aligned M1 partitions found. Run `slytrade align` first.")
        return ScanResult(signals=[], rows_scanned=0)

    files = sorted(files)
    need_cols = list(dict.fromkeys(_strategy_columns(cfg)))

    signals: List[Signal] = []
    total_rows = 0

    warmup_n = 200  # rows carried across month boundary (zone-state continuity)
    prev_tail = pd.DataFrame()
    state: Dict = {}  # running zone + trigger state across partitions

    for f_idx, f in enumerate(files):
        try:
            pf = pq.ParquetFile(str(f))
            have = set(pf.schema.names)
            cols = [c for c in need_cols if c in have]
            chunk = pd.read_parquet(f, columns=cols)
        except Exception as e:
            progress(f"  skip {f.name}: {e}")
            continue

        chunk["time"] = pd.to_datetime(chunk["time"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)
        if not prev_tail.empty:
            combined = pd.concat([prev_tail, chunk], ignore_index=True)
        else:
            combined = chunk
        n = len(combined)
        start_i = len(prev_tail)
        total_rows += len(chunk)

        # --- Vectorized pre-filter: rows worth running the full checklist on ---
        close = _col(combined, 'close')
        atr = _col(combined, 'atr_14')
        atr_pct = np.where(close > 0, atr / close, 0.0)
        session = combined['session'].to_numpy() if 'session' in combined.columns else np.full(n, 'OFF')
        kz_lon = _bcol(combined, 'kz_london')
        kz_ny  = _bcol(combined, 'kz_ny')
        kz_as  = _bcol(combined, 'kz_asian')
        lon_o30 = _bcol(combined, 'london_open_30')
        ny_o30  = _bcol(combined, 'ny_open_30')

        off = (session == 'OFF')
        allowed = ((cfg.sessions.trade_london_kz & kz_lon) |
                   (cfg.sessions.trade_london_open30 & lon_o30) |
                   (cfg.sessions.trade_ny_kz & kz_ny) |
                   (cfg.sessions.trade_ny_open30 & ny_o30) |
                   (cfg.sessions.trade_asian_range_retest & kz_as))
        if cfg.sessions.block_off_hours:
            allowed &= ~off

        atr_ok = (atr_pct >= cfg.confluence.min_atr_pct) & (atr_pct <= cfg.confluence.max_atr_pct) & (atr > 0)

        # Loose pre-filter: ATR + session. Zone/trigger timing is enforced
        # inside _evaluate_row via the state dict so we catch retests that
        # happen N bars after a displacement (not just on the disp bar itself).
        candidate_mask = allowed & atr_ok
        candidate_idx = np.nonzero(candidate_mask[start_i:])[0] + start_i

        # Advance state across warmup rows first (so zone mitigations stay
        # consistent at the month boundary) without emitting signals.
        for i in range(0, start_i):
            try:
                _evaluate_row(int(i), combined.iloc[i], cfg, state)
            except Exception:
                pass

        month_signals = 0
        for i in candidate_idx:
            row = combined.iloc[i]
            try:
                sig = _evaluate_row(int(i), row, cfg, state)
            except Exception:
                sig = None
            if sig is not None:
                signals.append(sig)
                month_signals += 1

        prev_tail = combined.tail(warmup_n).reset_index(drop=True)
        progress(f"  {f.parent.name}/{f.name}: {len(chunk):,} rows, eval={candidate_mask[start_i:].sum()}, signals={month_signals} (total={len(signals):,})")

    # Final dedupe (same direction within 5 M1 bars = same setup)
    signals.sort(key=lambda s: s.time)
    grade_rank = {'A+':4,'A':3,'B':2,'C':1}
    deduped: List[Signal] = []
    for s in signals:
        if deduped and (s.time - deduped[-1].time) < pd.Timedelta(minutes=5) and s.direction == deduped[-1].direction:
            if grade_rank.get(s.grade,0) > grade_rank.get(deduped[-1].grade,0):
                deduped[-1] = s
            continue
        deduped.append(s)

    progress(f"Scan complete: {total_rows:,} rows, {len(deduped):,} signals.")
    return ScanResult(signals=deduped, rows_scanned=total_rows)


# --------------------------------------------------------------------------- #
# Persist signals to parquet (data/processed/signals/)
# --------------------------------------------------------------------------- #

def write_signals(data_cfg: DataConfig,
                  raw_symbol: str,
                  signals: List[Signal]) -> Path:
    """Write signal list as a single parquet under data/processed/signals/."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .signals import signals_to_frame

    out_dir = ensure_dir(data_cfg.processed_root / "signals" / f"symbol={raw_symbol}")
    out = out_dir / "signals.parquet"
    sdf = signals_to_frame(signals)
    if sdf.empty:
        # Write empty schema
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({}), out)
        return out
    sdf['time'] = pd.to_datetime(sdf['time'], utc=True, errors='coerce')
    sdf = _normalize_for_parquet(sdf, time_col='time')
    _atomic_write_parquet(sdf, out)
    return out


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
