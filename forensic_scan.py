"""Deep forensic scalping analysis — Sly becomes the market.

Instead of guessing config values, I run the signal engine with maximally
permissive gates (capture EVERY potential setup), then measure what actually
happens to price in the N bars after each setup. This tells us:
  * Optimal TP (how far does price go in our direction 90%/70%/50% of the time)
  * Optimal SL (where do setups that fail actually get taken out)
  * Which hours/days/zone-types/grades are consistently winners
  * Whether partial exits / BE locks improve PF
  * Win rate × R:R tradeoffs along the PF frontier
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import replace

# Add project
sys.path.insert(0, str(Path(__file__).parent / "src"))

from slytrade.config import DataConfig
from slytrade.strategy.config import StrategyConfig, ConfluenceConfig, ExitPlan, SetupGrades, SessionFilter
from slytrade.strategy.signals import Signal, _evaluate_row, _strategy_columns
import pyarrow.parquet as pq


def load_all_signals(permissive=True):
    """Generate signals with maximally permissive gates to capture ALL setups."""
    cfg = DataConfig.from_paths(Path("data/raw"), Path("data/processed"))
    base = cfg.aligned_path / "symbol=XAUUSDm" / "timeframe=M1"
    files = sorted(base.glob("**/*.parquet"))
    print(f"Loading {len(files)} aligned partitions...")

    # Build permissive config that captures everything
    if permissive:
        cc = ConfluenceConfig(
            require_trigger_bos_or_choch=True,
            ob_tfs=("H1","M15","M5"),
            pd_range_tf="M15",
            pd_zone_min_pct=0.0,
            pd_zone_max_pct=1.0,  # accept anywhere
            a_plus_required_tfs=("D1","H4","H1"),
            a_required_tfs=("H4","H1"),
            b_required_tfs=("H1","M15"),
            c_required_tfs=("M15",),
            killzone_confluence_bonus=True,
            min_atr_pct=0.0,       # no atr floor
            max_atr_pct=1.0,       # no atr ceiling
            min_risk_atr=0.0,      # no min risk width
            max_risk_atr=999.0,    # no max risk width
            accept_ob_tfs=("H1","M15","M5"),  # accept all OB TFs
            accept_zone_kinds=("OB","FVG"),   # accept both OB and FVG
            accept_grades=("A+","A","B","C"), # accept all grades
        )
        sf = SessionFilter(
            trade_london_kz=True, trade_ny_kz=True,
            trade_asian_kz=True, trade_london_open30=True,
            trade_ny_open30=True, trade_asian_range_retest=True,
            block_off_hours=False,  # accept off-hours too
        )
        scfg = StrategyConfig(
            grades=SetupGrades(),
            exits=ExitPlan(tp1_r=0.5, tp1_pct=1.0, tp2_r=1.0, tp2_pct=0.0,
                          ob_invalidation_buffer=0.05, time_stop_bars=240),
            sessions=sf, confluence=cc,
        )
    else:
        scfg = StrategyConfig()

    need_cols = list(dict.fromkeys(_strategy_columns(scfg)))
    # Make sure we have spread
    if "spread" not in need_cols: need_cols.append("spread")
    if "tick_volume" not in need_cols: need_cols.append("tick_volume")

    all_sigs = []
    state = {}
    total_rows = 0
    warmup = 500
    rows_seen = 0
    for f_idx, f in enumerate(files):
        pf = pq.ParquetFile(str(f))
        have = set(pf.schema.names)
        cols = [c for c in need_cols if c in have]
        chunk = pd.read_parquet(f, columns=cols)
        chunk["time"] = pd.to_datetime(chunk["time"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)
        for i in range(len(chunk)):
            rows_seen += 1
            if rows_seen < warmup:
                continue
            sig = _evaluate_row(i, chunk.iloc[i], scfg, state)
            if sig is not None:
                sig_rec = {
                    "time": sig.time, "direction": sig.direction,
                    "entry": sig.entry, "stop": sig.stop,
                    "risk": sig.risk_per_unit, "grade": sig.grade,
                    "trigger_tf": sig.trigger_tf, "ob_tf": sig.ob_tf or "",
                    "zone_kind": _zone_kind(sig),
                    "session": sig.session, "killzone": sig.killzone,
                    "atr": sig.atr_at_entry,
                    "fvg_top": sig.fvg_top, "fvg_bottom": sig.fvg_bottom,
                    "ob_top": sig.ob_top, "ob_bottom": sig.ob_bottom,
                }
                # Add HTF bias
                for tf in ("W1","D1","H4","H1","M30","M15","M5"):
                    sig_rec[f"bias_{tf}"] = int(sig.htf_bias_summary.get(tf,0) or 0)
                all_sigs.append(sig_rec)
        total_rows += len(chunk)
        if (f_idx+1) % 5 == 0 or f_idx == len(files)-1:
            print(f"  {f.name}: rows={total_rows:,} signals={len(all_sigs):,}")

    sig_df = pd.DataFrame(all_sigs)
    print(f"\nTotal candidate signals: {len(sig_df):,}")
    return sig_df, files, need_cols


def _zone_kind(sig):
    ob = getattr(sig, "ob_tf", None)
    if ob is None: return "FVG"
    try:
        import pandas as pd
        if pd.isna(ob): return "FVG"
    except: pass
    return "OB" if ob else "FVG"


if __name__ == "__main__":
    sig_df, files, need_cols = load_all_signals(permissive=True)
    sig_df.to_parquet("/tmp/all_candidate_signals.parquet", index=False)
    print("Saved to /tmp/all_candidate_signals.parquet")
    print(f"\nGrade distribution:\n{sig_df['grade'].value_counts()}")
    print(f"\nZone kind:\n{sig_df['zone_kind'].value_counts()}")
    print(f"\nOB TF:\n{sig_df['ob_tf'].replace('', 'FVG').value_counts()}")
    print(f"\nKillzone:\n{sig_df['killzone'].value_counts()}")
    print(f"\nSession:\n{sig_df['session'].value_counts()}")
    print(f"\nDirection:\n{sig_df['direction'].value_counts()}")
