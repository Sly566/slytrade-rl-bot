"""The M5 edge — deep dive on M5 OB longs which showed PF 3.32."""
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
    t0 = time.time()
    bars_df, sdf, _ = load_bars_and_signals()
    arr = build_bars_arr(bars_df)
    spec = spec_for_symbol("XAUUSDm")
    acct = AccountSpec(starting_equity=20000, currency="ZAR", leverage=2000, fx_to_account={"USD":18.5})
    bt = BacktestConfig(starting_equity=20000, pay_entry_spread=True,
                        slippage_points_long=5, slippage_points_short=5,
                        max_open_positions=10, min_equity_fraction=0.30,
                        max_risk_per_trade=0.02, commission_per_lot_rt=0.0)

    s = sdf.copy()
    s["hour_utc"] = s["time"].dt.hour
    s["dow"] = s["time"].dt.dayofweek
    s["risk_atr"] = abs(s["entry"] - s["stop"]) / s["atr_at_entry"]
    s["dir_int"] = s["direction"].astype(int)

    def run(mask, scfg, label):
        fsig = s[mask]
        if len(fsig) < 5:
            print(f"  {label:65s} SKIP n={len(fsig)}")
            return None
        sidx = build_sig_index(fsig, arr["time"])
        r = run_fast(arr, sidx, spec, acct, bt, scfg)
        m = calc_metrics(r.trades)
        star = " ***" if m["profit_factor"]>=2.0 and m["n_trades"]>=30 else (" *" if m["profit_factor"]>=1.8 else "")
        print(f"  {fmt(m, label)}{star}")
        return m, r

    # ===== THE DISCOVERY: M5 OB IS WHERE THE MONEY IS =====
    print("=== M5 vs M15 comparison (LONGS) ===")
    for label, mask in [
        ("LONG ALL OB", (s.dir_int==1)),
        ("LONG M15 OB only", (s.dir_int==1)&(s.ob_tf=="M15")),
        ("LONG M5 OB only", (s.dir_int==1)&(s.ob_tf=="M5")),
    ]:
        run(mask, make_cfg(), label)

    print("\n=== M5 OB LONG × tp1_r × min_risk ATR ===")
    results = []
    for tp in [0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,1.0]:
      for min_r in [0.0,1.2,1.5,2.0,2.5,3.0]:
        mask = (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=min_r)
        m,_ = run(mask, make_cfg(tp1_r=tp, min_risk=min_r, ob_tfs=("M5",)),
                  f"M5 LONG tp={tp:.2f} r>={min_r:.1f}")
        if m: results.append((m["profit_factor"], m["n_trades"], m["total_pnl"], m["mean_R"], tp, min_r))

    print("\n=== TOP 15 M5 LONG configs (sorted by PF, n>=25) ===")
    top = sorted([r for r in results if r[1]>=25], key=lambda x: -x[0])[:15]
    for pf, n, pnl, mR, tp, mr in top:
        print(f"  tp={tp:.2f} r>={mr:.1f}  n={n:3d}  PF={pf:.2f}  mR={mR:+.3f}  PnL={pnl:+,.0f}")

    print("\n=== M5 OB LONG × grades ===")
    for grades, gl in [
        (("A+","A","B","C"), "ALL+C"),
        (("A+","A","B"), "ALL"),
        (("A+","A"), "A+/A"),
        (("A+",), "A+ only"),
        (("A+","A","B","C"), "ALL+C (r>=2.0)"),
    ]:
        extra = (s.risk_atr>=2.0) if "r>=2.0" in gl else True
        run((s.dir_int==1)&(s.ob_tf=="M5")&s.grade.isin(grades)&extra,
            make_cfg(tp1_r=0.75, ob_tfs=("M5",), grades=grades, min_risk=2.0 if "r>=2.0" in gl else 0.0),
            f"M5 LONG tp=0.75 {gl}")

    print("\n=== M5 LONG × hour windows (tp=0.75 r>=2.0) ===")
    for (lo, hi) in [(7,16),(8,16),(9,16),(7,15),(8,15),(9,15),(10,16),(8,17),(7,13),(8,13),(9,13),(10,15)]:
        mask = (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.0)&s.hour_utc.between(lo,hi)
        run(mask, make_cfg(tp1_r=0.75, min_risk=2.0, ob_tfs=("M5",)),
            f"M5 LONG tp=0.75 r>=2 UTC {lo:02d}-{hi:02d}")

    print("\n=== M5 LONG × day of week (tp=0.75 r>=2.0) ===")
    dow_names = ["Mon","Tue","Wed","Thu","Fri"]
    for d in range(5):
        mask = (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.0)&(s.dow==d)
        run(mask, make_cfg(tp1_r=0.75, min_risk=2.0, ob_tfs=("M5",)),
            f"M5 LONG tp=0.75 r>=2 {dow_names[d]}")

    print("\n=== WALK-FORWARD on M5 LONG configs ===")
    split_time = pd.Timestamp("2025-08-01", tz="UTC")
    train_mask = (s.time < split_time)
    test_mask = ~train_mask
    for label, mask, cfg in [
        ("M5 LONG tp=0.75 r>=2.0", (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.0),
            make_cfg(tp1_r=0.75, min_risk=2.0, ob_tfs=("M5",))),
        ("M5 LONG tp=0.75 all", (s.dir_int==1)&(s.ob_tf=="M5"),
            make_cfg(tp1_r=0.75, ob_tfs=("M5",))),
        ("M5 LONG tp=0.85 r>=2.5", (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.5),
            make_cfg(tp1_r=0.85, min_risk=2.5, ob_tfs=("M5",))),
        ("M5 LONG tp=0.6 r>=2.0", (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.0),
            make_cfg(tp1_r=0.6, min_risk=2.0, ob_tfs=("M5",))),
        ("M5 LONG tp=0.75 A+/A r>=2.0", (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.0)&s.grade.isin(["A+","A"]),
            make_cfg(tp1_r=0.75, min_risk=2.0, ob_tfs=("M5",), grades=("A+","A"))),
        # Compare M5 vs ALL on the champion tp=0.85 r>=2.5
        ("ALL LONG tp=0.85 r>=2.5", (s.dir_int==1)&(s.risk_atr>=2.5),
            make_cfg(tp1_r=0.85, min_risk=2.5)),
    ]:
        fsig_tr = s[mask & train_mask]; fsig_te = s[mask & test_mask]
        if len(fsig_tr)<8 or len(fsig_te)<8:
            print(f"  {label:40s} train n={len(fsig_tr)} test n={len(fsig_te)} SKIP")
            continue
        r_tr = run_fast(arr, build_sig_index(fsig_tr, arr["time"]), spec, acct, bt, cfg)
        r_te = run_fast(arr, build_sig_index(fsig_te, arr["time"]), spec, acct, bt, cfg)
        m_tr = calc_metrics(r_tr.trades); m_te = calc_metrics(r_te.trades)
        deg = "DEGRADED" if m_te["profit_factor"] < 0.7*m_tr["profit_factor"] else ("OK" if m_te["profit_factor"]>=1.5 else "WEAK")
        print(f"  {label:40s}  TR n={m_tr['n_trades']:3d} PF={m_tr['profit_factor']:.2f} wr={m_tr['win_rate']*100:.1f}% PnL={m_tr['total_pnl']:+,.0f}"
              f"  | TE n={m_te['n_trades']:3d} PF={m_te['profit_factor']:.2f} wr={m_te['win_rate']*100:.1f}% PnL={m_te['total_pnl']:+,.0f}  [{deg}]")

    # ZAR 1000 MICRO
    print("\n=== ZAR 1000 MICRO — M5 configs ===")
    acct1k = AccountSpec(starting_equity=1000, currency="ZAR", leverage=2000, fx_to_account={"USD":18.5})
    bt1k = BacktestConfig(starting_equity=1000, pay_entry_spread=True,
                          slippage_points_long=5, slippage_points_short=5,
                          max_open_positions=3, min_equity_fraction=0.30,
                          max_risk_per_trade=0.015, commission_per_lot_rt=0.0)
    for label, mask, cfg in [
        ("BASELINE", pd.Series(True,index=s.index), make_cfg()),
        ("M5 LONG tp=0.75 r>=2.0", (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.0),
            make_cfg(tp1_r=0.75, min_risk=2.0, ob_tfs=("M5",))),
        ("M5 LONG tp=0.6 r>=2.0", (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.0),
            make_cfg(tp1_r=0.6, min_risk=2.0, ob_tfs=("M5",))),
        ("M5 LONG tp=0.75 r>=2.0 max3 1.5%", (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.0),
            make_cfg(tp1_r=0.75, min_risk=2.0, ob_tfs=("M5",))),
        ("M5 LONG tp=0.5 r>=2.0", (s.dir_int==1)&(s.ob_tf=="M5")&(s.risk_atr>=2.0),
            make_cfg(tp1_r=0.5, min_risk=2.0, ob_tfs=("M5",))),
    ]:
        fsig = s[mask]
        sidx = build_sig_index(fsig, arr["time"])
        r = run_fast(arr, sidx, spec, acct1k, bt1k, cfg)
        m = calc_metrics(r.trades, equity=1000)
        print(f"  {fmt(m, label)}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
