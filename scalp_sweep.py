"""Targeted sweep around the LONG + wide-stop edge to find optimal config."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import pandas as pd
from scalp_fast import (load_bars_and_signals, build_bars_arr, build_sig_index,
                         run_fast, calc_metrics, fmt)
from slytrade.backtest.specs import spec_for_symbol, AccountSpec
from slytrade.backtest.engine import BacktestConfig
from slytrade.strategy.config import StrategyConfig, ExitPlan, SessionFilter, ConfluenceConfig


def make_cfg(tp1_r=0.75, tp1_pct=1.0, tp2_r=1.5, tp2_pct=0.0,
             grades=("A+","A","B"), ob_tfs=("M15","M5"), zones=("OB",),
             min_risk=0.0, max_risk=99.0, atr_trail=0.5,
             time_stop=240, time_stop_min_r=0.0, be_buffer=0.05):
    return StrategyConfig(
        exits=ExitPlan(tp1_r=tp1_r, tp1_pct=tp1_pct, tp2_r=tp2_r, tp2_pct=tp2_pct,
                       ob_invalidation_buffer=be_buffer, runner_trail_atr_mult=atr_trail,
                       time_stop_bars=time_stop, time_stop_min_r=time_stop_min_r),
        confluence=ConfluenceConfig(accept_grades=grades, accept_ob_tfs=ob_tfs, accept_zone_kinds=zones,
                                    min_risk_atr=min_risk, max_risk_atr=max_risk),
        sessions=SessionFilter(block_off_hours=True, trade_asian_kz=False, trade_asian_range_retest=False))


def main():
    t0=time.time()
    bars_df, sdf, _ = load_bars_and_signals()
    arr = build_bars_arr(bars_df)
    spec = spec_for_symbol("XAUUSDm")
    acct = AccountSpec(starting_equity=20000, currency="ZAR", leverage=2000, fx_to_account={"USD":18.5})
    bt = BacktestConfig(starting_equity=20000, pay_entry_spread=True, slippage_points_long=5, slippage_points_short=5,
                        max_open_positions=10, min_equity_fraction=0.30, max_risk_per_trade=0.02)
    s = sdf.copy()
    s["hour_utc"] = s["time"].dt.hour
    s["dow"] = s["time"].dt.dayofweek
    s["risk_atr"] = abs(s["entry"]-s["stop"])/s["atr_at_entry"]
    s["dir_int"] = s["direction"].astype(int)

    def run(mask, scfg, label):
        fsig = s[mask]
        if len(fsig) < 10: return None
        sidx = build_sig_index(fsig, arr["time"])
        r = run_fast(arr, sidx, spec, acct, bt, scfg)
        m = calc_metrics(r.trades)
        return m

    # Key insight: LONGS are massively better. Let's explore LONG + risk>=X + tp=Y
    print("=== LONGS: tp × risk-ATR grid ===")
    results = []
    for tp1 in [0.4,0.5,0.6,0.75,1.0]:
      for min_r in [1.2,1.5,2.0,2.5,3.0,3.5]:
        mask = (s.dir_int==1) & (s.risk_atr>=min_r)
        m = run(mask, make_cfg(tp1_r=tp1, min_risk=min_r), f"LONG tp={tp1} risk>={min_r}")
        if m:
          results.append((m["profit_factor"], m["n_trades"], m["total_pnl"], m["max_drawdown_pct"],
                          tp1, min_r, "ALL_HOURS"))
          star = " ***" if m["profit_factor"]>=1.8 and m["n_trades"]>=40 else ""
          print(f"  {fmt(m, f'LONG tp={tp1} risk>={min_r}')}{star}")

    print("\n=== LONGS risk>=X × HOUR FILTER (8-15 Lon+NY) ===")
    for tp1 in [0.5,0.6,0.75]:
      for min_r in [2.0,2.5,3.0]:
        for (hlo,hhi,lbl) in [(8,15,"8-15"),(7,15,"7-15"),(8,14,"8-14"),(12,15,"12-15"),(None,None,"all")]:
          m2 = (s.dir_int==1)&(s.risk_atr>=min_r)
          lab = "all"
          if hlo is not None:
            m2 &= s.hour_utc.between(hlo,hhi); lab=f"{hlo}-{hhi}"
          m = run(m2, make_cfg(tp1_r=tp1, min_risk=min_r), f"LONG {lab} tp={tp1} risk>={min_r}")
          if m:
            star = " ***" if m["profit_factor"]>=2.0 and m["n_trades"]>=40 else ""
            print(f"  {fmt(m, f'LONG {lab:5s} tp={tp1} risk>={min_r}')}{star}")
            results.append((m["profit_factor"], m["n_trades"], m["total_pnl"], m["max_drawdown_pct"], tp1, min_r, lab))

    print("\n=== LONGS × GRADES × M15 only ===")
    for grades, glabel in [(("A+","A","B"),"all"),(("A+","A"),"A+/A"),(("A+",),"A+")]:
      for ob_tfs, olabel in [(("M15","M5"),"M15+M5"),(("M15",),"M15")]:
        for tp1 in [0.6,0.75]:
          for min_r in [2.0,2.5,3.0]:
            mask = (s.dir_int==1)&(s.risk_atr>=min_r)&s.grade.isin(grades)&(s.ob_tf.isin(ob_tfs) if "M5" not in olabel else True)
            if olabel=="M15": mask &= s.ob_tf=="M15"
            m = run(mask, make_cfg(tp1_r=tp1, min_risk=min_r, grades=grades, ob_tfs=ob_tfs),
                    f"LONG {glabel:5s} {olabel} tp={tp1} risk>={min_r}")
            if m:
              star = " ***" if m["profit_factor"]>=2.0 and m["n_trades"]>=30 else ""
              print(f"  {fmt(m, f'LONG {glabel:5s} {olabel} tp={tp1} risk>={min_r}')}{star}")

    print("\n=== BOTH DIRECTIONS but LONGS ONLY after filtering shorts with strict gates ===")
    # What if we trade SHORTS only during specific hours where they work?
    print("\n=== SHORTS by hour ===")
    for (hlo,hhi,lbl) in [(7,10,"7-10"),(12,15,"12-15"),(13,15,"13-15"),(8,9,"8-9"),(None,None,"all")]:
      for min_r in [1.2,2.0,3.0]:
        m2 = (s.dir_int==-1)&(s.risk_atr>=min_r)
        lab = "all"
        if hlo is not None: m2 &= s.hour_utc.between(hlo,hhi); lab=f"{hlo}-{hhi}"
        m = run(m2, make_cfg(tp1_r=0.75, min_risk=min_r), f"SHORT {lab} risk>={min_r}")
        if m and m["n_trades"]>=10:
          print(f"  {fmt(m, f'SHORT {lab:5s} tp=0.75 risk>={min_r}')}")

    # Micro account test (ZAR 1000 with min-lot floor)
    print("\n=== ZAR 1000 MICRO ACCOUNT on best LONG config ===")
    for tp1, min_r in [(0.6, 2.5), (0.75, 2.5), (0.6, 3.0), (0.75, 3.0)]:
      mask = (s.dir_int==1)&(s.risk_atr>=min_r)
      fsig = s[mask]
      sidx = build_sig_index(fsig, arr["time"])
      acct_micro = AccountSpec(starting_equity=1000, currency="ZAR", leverage=2000, fx_to_account={"USD":18.5})
      bt_micro = BacktestConfig(starting_equity=1000, max_risk_per_trade=0.02,
                                slippage_points_long=5, slippage_points_short=5,
                                pay_entry_spread=True, max_open_positions=5, min_equity_fraction=0.30)
      r = run_fast(arr, sidx, spec, acct_micro, bt_micro, make_cfg(tp1_r=tp1, min_risk=min_r))
      m = calc_metrics(r.trades, equity=1000)
      print(f"  {fmt(m, f'ZAR1000 LONG tp={tp1} risk>={min_r}')}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
