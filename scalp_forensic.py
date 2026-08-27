"""Deep scalping forensic analysis — find what really moves the needle.

Key insight: with 388 signals we only have ~380 trades. To push PF to 1.8-2.0,
we need to eliminate the ~150 losing trades while keeping as many winners as
possible. I'm going to measure every possible conditioning variable and
find the cut points that filter losses while preserving wins.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from dataclasses import replace

from slytrade.backtest import run_backtest, BacktestConfig, AccountSpec
from slytrade.config import DataConfig
from slytrade.strategy.config import (
    StrategyConfig, SetupGrades, ExitPlan, SessionFilter, ConfluenceConfig
)


def run_with_config(signals_df, scfg, equity=20000, slip=5, max_risk=0.02, spreads=True, usd_zar=18.5):
    cfg = DataConfig.from_paths(Path("data/raw"), Path("data/processed"))
    acct = AccountSpec(starting_equity=equity, currency="ZAR", leverage=2000,
                       fx_to_account={"USD": usd_zar}, commission_per_lot_rt=0.0)
    bt_cfg = BacktestConfig(starting_equity=equity, account_ccy="ZAR", leverage=2000,
                            usd_zar=usd_zar, slippage_points_long=slip, slippage_points_short=slip,
                            commission_per_lot_rt=0.0, pay_entry_spread=spreads,
                            max_open_positions=10, min_equity_fraction=0.30, max_risk_per_trade=max_risk)
    result = run_backtest(cfg, "XAUUSDm", signals_df, account=acct, bt_cfg=bt_cfg, strat_cfg=scfg,
                          progress=lambda m: None)
    return result


def metrics_for(result):
    m = result.metrics
    return {
        "n": m["n_trades"], "wr": m["win_rate"], "pf": m["profit_factor"],
        "meanR": m["mean_R"], "pnl": m["total_pnl"],
        "ret": m.get("return_pct", 0), "dd": m.get("max_drawdown_pct", 0),
    }


def fmt(m):
    return f"n={m['n']:4d} wr={m['wr']*100:5.1f}% PF={m['pf']:.2f} mR={m['meanR']:+.3f} PnL={m['pnl']:+,.0f} ret={m['ret']:+.1f}% dd={m['dd']:.1f}%"


def main():
    cfg = DataConfig.from_paths(Path("data/raw"), Path("data/processed"))
    sdf = pd.read_parquet(cfg.signals_path / "symbol=XAUUSDm/signals.parquet")
    sdf["time"] = pd.to_datetime(sdf["time"], utc=True)
    print(f"Loaded {len(sdf)} baseline signals\n")

    # BASELINE
    base = run_with_config(sdf, StrategyConfig())
    print("BASELINE:", fmt(metrics_for(base)))
    print()

    # Save baseline trades to dissect
    trades = base.trades
    trades.to_parquet("/tmp/baseline_trades.parquet")
    print(f"Trades: {len(trades)}")
    print(f"\n=== EXIT REASON DEEP DIVE ===")
    print(trades.groupby("close_reason").agg(
        n=("pos_id", "count"),
        win=("pnl_acct", lambda x: (x>0).sum()),
        meanR=("r_multiple", "mean"),
        pnl=("pnl_acct", "sum"),
    ).to_string())

    print(f"\n=== HOUR-OF-DAY ANALYSIS (entry time UTC -> Johannesburg is UTC+2) ===")
    trades["hour_utc"] = trades["entry_time"].dt.hour
    trades["dow"] = trades["entry_time"].dt.dayofweek  # 0=Mon
    hod = trades.groupby("hour_utc").agg(
        n=("pos_id","count"), win=("pnl_acct",lambda x:(x>0).sum()),
        meanR=("r_multiple","mean"), pnl=("pnl_acct","sum"),
        pf=("pnl_acct", lambda x: x[x>0].sum()/max(-x[x<0].sum(),1e-9)))
    print(hod.to_string())

    print(f"\n=== DAY OF WEEK ===")
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    trades["dow_name"] = trades["dow"].map(dow_map)
    dwd = trades.groupby("dow_name").agg(
        n=("pos_id","count"), wr=("pnl_acct",lambda x:(x>0).mean()),
        meanR=("r_multiple","mean"), pnl=("pnl_acct","sum"),
        pf=("pnl_acct", lambda x: x[x>0].sum()/max(-x[x<0].sum(),1e-9)))
    print(dwd.reindex(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]).dropna().to_string())

    print(f"\n=== HOUR × DIRECTION ===")
    hd = trades.groupby(["hour_utc","direction"]).agg(
        n=("pos_id","count"),
        pf=("pnl_acct", lambda x: x[x>0].sum()/max(-x[x<0].sum(),1e-9)),
        pnl=("pnl_acct","sum"),
        meanR=("r_multiple","mean"))
    print(hd.to_string())

    print(f"\n=== ATR-AT-ENTRY BUCKETS ===")
    sig_small = sdf[["time","direction","entry","atr_at_entry"]].copy()
    sig_small["risk_atr"] = abs(sdf["entry"] - sdf["stop"]) / sdf["atr_at_entry"]
    sig_small["atr_bin"] = pd.cut(sdf["atr_at_entry"], bins=[0,5,10,15,20,25,30,40,60,100])
    sig_small["risk_bin"] = pd.cut(sig_small["risk_atr"], bins=[0,1.0,1.2,1.5,2.0,2.5,3.0,4.0,6.0])
    # trades direction is "LONG"/"SHORT" strings; convert
    merged = trades.copy()
    merged["dir_int"] = (merged["direction"] == "LONG").astype(int) * 2 - 1
    merged = merged.merge(sig_small, left_on=["entry_time","dir_int"], right_on=["time","direction"], how="left", suffixes=("","_s"))

    ab = merged.groupby("atr_bin", observed=True).agg(
        n=("pos_id","count"),
        wr=("pnl_acct", lambda x:(x>0).mean()),
        pf=("pnl_acct", lambda x: x[x>0].sum()/max(-x[x<0].sum(),1e-9)),
        meanR=("r_multiple","mean"), pnl=("pnl_acct","sum"))
    print(ab.to_string())

    # Risk/ATR ratio bucket
    print(f"\n=== RISK-IN-ATR BUCKETS ===")
    rb = merged.groupby("risk_bin", observed=True).agg(
        n=("pos_id","count"), wr=("pnl_acct",lambda x:(x>0).mean()),
        pf=("pnl_acct",lambda x: x[x>0].sum()/max(-x[x<0].sum(),1e-9)),
        meanR=("r_multiple","mean"), pnl=("pnl_acct","sum"))
    print(rb.to_string())

    # Killzone × grade × direction
    print(f"\n=== KILLZONE × GRADE ===")
    kg = trades.groupby(["killzone","grade"]).agg(
        n=("pos_id","count"),
        pf=("pnl_acct",lambda x: x[x>0].sum()/max(-x[x<0].sum(),1e-9)),
        meanR=("r_multiple","mean"), pnl=("pnl_acct","sum"))
    print(kg.to_string())

    # OB/FVG zone TF analysis
    print(f"\n=== OB_TF × GRADE ===")
    ot = trades.groupby(["ob_tf","grade"]).agg(
        n=("pos_id","count"),
        pf=("pnl_acct",lambda x: x[x>0].sum()/max(-x[x<0].sum(),1e-9)),
        meanR=("r_multiple","mean"), pnl=("pnl_acct","sum"))
    print(ot.to_string())

    # Bars held distribution (do quick scalps win more than long holds?)
    print(f"\n=== BARS HELD BUCKETS ===")
    trades["bars_bin"] = pd.cut(trades["bars_held"], bins=[0,5,15,30,60,120,240,500])
    bb = trades.groupby("bars_bin", observed=True).agg(
        n=("pos_id","count"), wr=("pnl_acct",lambda x:(x>0).mean()),
        pf=("pnl_acct",lambda x: x[x>0].sum()/max(-x[x<0].sum(),1e-9)),
        meanR=("r_multiple","mean"), pnl=("pnl_acct","sum"))
    print(bb.to_string())


if __name__ == "__main__":
    main()
