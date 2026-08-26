"""Deep forensic: day-of-week, session, walk-forward, and the quest for PF>2.0 at n>=100."""
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
             time_stop=240, time_stop_min_r=0.0, be_buffer=0.05,
             block_off=True, asian=False):
    return StrategyConfig(
        exits=ExitPlan(tp1_r=tp1_r, tp1_pct=tp1_pct, tp2_r=tp2_r, tp2_pct=tp2_pct,
                       ob_invalidation_buffer=be_buffer, runner_trail_atr_mult=atr_trail,
                       time_stop_bars=time_stop, time_stop_min_r=time_stop_min_r),
        confluence=ConfluenceConfig(accept_grades=grades, accept_ob_tfs=ob_tfs, accept_zone_kinds=zones,
                                    min_risk_atr=min_risk, max_risk_atr=max_risk),
        sessions=SessionFilter(block_off_hours=block_off, trade_asian_kz=asian, trade_asian_range_retest=False))


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
    s["dow"] = s["time"].dt.dayofweek  # Mon=0 ... Sun=6
    s["risk_atr"] = abs(s["entry"] - s["stop"]) / s["atr_at_entry"]
    s["dir_int"] = s["direction"].astype(int)
    s["sast_hour"] = (s["hour_utc"] + 2) % 24  # SAST = UTC+2
    s["month"] = s["time"].dt.to_period("M")

    def run(mask, scfg, label, verbose=True):
        fsig = s[mask]
        if len(fsig) < 5:
            if verbose: print(f"  {label:60s} SKIP n={len(fsig)}")
            return None
        sidx = build_sig_index(fsig, arr["time"])
        r = run_fast(arr, sidx, spec, acct, bt, scfg)
        m = calc_metrics(r.trades)
        if verbose:
            star = " ***" if m["profit_factor"]>=2.0 and m["n_trades"]>=40 else (" *" if m["profit_factor"]>=1.8 and m["n_trades"]>=30 else "")
            print(f"  {fmt(m, label)}{star}")
        return m, r

    # ======================== DAY OF WEEK ========================
    print("=== LONGS tp=0.6 risk>=2.0 BY DAY OF WEEK (UTC) ===")
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for d in range(7):
        mask = (s.dir_int==1)&(s.risk_atr>=2.0)&(s.dow==d)
        run(mask, make_cfg(tp1_r=0.6, min_risk=2.0), f"  LONG tp=0.6 r>=2 {dow_names[d]}")
    print()
    print("=== LONGS tp=0.75 risk>=2.5 BY DAY OF WEEK ===")
    for d in range(7):
        mask = (s.dir_int==1)&(s.risk_atr>=2.5)&(s.dow==d)
        run(mask, make_cfg(tp1_r=0.75, min_risk=2.5), f"  LONG tp=0.75 r>=2.5 {dow_names[d]}")

    # ======================== FINE HOUR WINDOWS ========================
    print("\n=== LONGS tp=0.6 r>=2.0 FINE HOUR WINDOWS (UTC) ===")
    for (lo, hi) in [(7,16),(8,16),(9,16),(7,15),(8,15),(9,15),(8,17),(9,17),
                     (8,12),(9,12),(10,16),(7,13),(8,13),(9,13),(10,15)]:
        mask = (s.dir_int==1)&(s.risk_atr>=2.0)&s.hour_utc.between(lo,hi)
        run(mask, make_cfg(tp1_r=0.6, min_risk=2.0), f"  LONG tp=0.6 r>=2  UTC {lo:02d}-{hi:02d}")

    print("\n=== LONGS tp=0.75 r>=2.5 FINE HOUR WINDOWS (UTC) ===")
    for (lo, hi) in [(7,16),(8,16),(9,16),(7,15),(8,15),(9,15),(8,17),(9,17),
                     (8,12),(9,12),(10,16),(7,13),(8,13),(9,13),(10,15)]:
        mask = (s.dir_int==1)&(s.risk_atr>=2.5)&s.hour_utc.between(lo,hi)
        run(mask, make_cfg(tp1_r=0.75, min_risk=2.5), f"  LONG tp=0.75 r>=2.5 UTC {lo:02d}-{hi:02d}")

    # ======================== TP FINE GRID on LONG r>=2.5 ========================
    print("\n=== LONGS r>=2.5 FINE TP GRID ===")
    for tp in np.arange(0.45, 1.05, 0.05):
        mask = (s.dir_int==1)&(s.risk_atr>=2.5)
        m,_ = run(mask, make_cfg(tp1_r=float(tp), min_risk=2.5), f"  LONG r>=2.5 tp={tp:.2f}R")

    # ======================== GRADES ON TOP CONFIGS ========================
    print("\n=== LONGS r>=2.5 × GRADES × OB TF ===")
    for tp in [0.6, 0.75]:
      for grades, gl in [(("A+","A","B"),"ALL"), (("A+","A"),"A+/A"), (("A+",),"A+"),
                         (("A+","A","B","C"),"ALL+C")]:
        for ob_tfs, ol in [(("M15","M5"),"ALL"), (("M15",),"M15"), (("M5",),"M5")]:
            mask = (s.dir_int==1)&(s.risk_atr>=2.5)&s.grade.isin(grades)
            if ol == "M15": mask &= s.ob_tf=="M15"
            elif ol == "M5": mask &= s.ob_tf=="M5"
            run(mask, make_cfg(tp1_r=tp, min_risk=2.5, grades=grades, ob_tfs=ob_tfs),
                f"  LONG tp={tp} r>=2.5 {gl:5s} {ol:4s}")

    # ======================== LADDERED EXITS on top long config ========================
    print("\n=== LADDERED EXITS: LONG r>=2.5 ===")
    for tp1, tp1pct, tp2, tp2pct in [
        (0.5, 0.5, 1.0, 0.3), (0.5, 0.5, 1.0, 0.5),
        (0.4, 0.5, 0.8, 0.3), (0.5, 0.6, 1.0, 0.2),
        (0.6, 0.5, 1.2, 0.3), (0.6, 0.6, 1.2, 0.2),
        (0.5, 1.0, 1.5, 0.0),  # one-shot control
    ]:
        mask = (s.dir_int==1)&(s.risk_atr>=2.5)
        run(mask, make_cfg(tp1_r=tp1, tp1_pct=tp1pct, tp2_r=tp2, tp2_pct=tp2pct, min_risk=2.5),
            f"  LONG ladder T1={tp1}@{tp1pct} T2={tp2}@{tp2pct}")

    # ======================== WALK-FORWARD: TRAIN/TEST SPLIT ========================
    print("\n=== WALK-FORWARD (train: Aug2024-Jul2025 / test: Aug2025-Aug2026) ===")
    # Split at ~60% mark by time
    split_time = pd.Timestamp("2025-08-01", tz="UTC")
    train_mask = (s.time < split_time)
    test_mask = ~train_mask
    print(f"  Train: before {split_time.date()}, n={int(train_mask.sum())} signals")
    print(f"  Test:  from   {split_time.date()}, n={int(test_mask.sum())} signals")

    for label, mask, cfg in [
        ("BASELINE 20k tp=0.75", pd.Series(True,index=s.index), make_cfg()),
        ("LONG tp=0.6 r>=2.0", (s.dir_int==1)&(s.risk_atr>=2.0), make_cfg(tp1_r=0.6, min_risk=2.0)),
        ("LONG tp=0.75 r>=2.5", (s.dir_int==1)&(s.risk_atr>=2.5), make_cfg(tp1_r=0.75, min_risk=2.5)),
        ("LONG tp=0.5 r>=2.0", (s.dir_int==1)&(s.risk_atr>=2.0), make_cfg(tp1_r=0.5, min_risk=2.0)),
        ("LONG tp=0.6 r>=2.5", (s.dir_int==1)&(s.risk_atr>=2.5), make_cfg(tp1_r=0.6, min_risk=2.5)),
        ("LONG tp=0.6 r>=2.0 A+/A M15", (s.dir_int==1)&(s.risk_atr>=2.0)&s.grade.isin(["A+","A"])&(s.ob_tf=="M15"),
            make_cfg(tp1_r=0.6, min_risk=2.0, grades=("A+","A"), ob_tfs=("M15",))),
        ("LONG tp=0.75 r>=2.0 A+/A", (s.dir_int==1)&(s.risk_atr>=2.0)&s.grade.isin(["A+","A"]),
            make_cfg(tp1_r=0.75, min_risk=2.0, grades=("A+","A"))),
        ("LONG tp=0.5 r>=2.5", (s.dir_int==1)&(s.risk_atr>=2.5), make_cfg(tp1_r=0.5, min_risk=2.5)),
    ]:
        # train
        fsig_tr = s[mask & train_mask]
        fsig_te = s[mask & test_mask]
        if len(fsig_tr)<10 or len(fsig_te)<10:
            print(f"  {label:45s} train n={len(fsig_tr)} test n={len(fsig_te)} SKIP")
            continue
        r_tr = run_fast(arr, build_sig_index(fsig_tr, arr["time"]), spec, acct, bt, cfg)
        r_te = run_fast(arr, build_sig_index(fsig_te, arr["time"]), spec, acct, bt, cfg)
        m_tr = calc_metrics(r_tr.trades); m_te = calc_metrics(r_te.trades)
        deg = "DEGRADED" if m_te["profit_factor"] < 0.8*m_tr["profit_factor"] else ("OK" if m_te["profit_factor"]>=1.5 else "WEAK")
        print(f"  {label:45s}  TR n={m_tr['n_trades']:3d} PF={m_tr['profit_factor']:.2f} wr={m_tr['win_rate']*100:.1f}% PnL={m_tr['total_pnl']:+,.0f}"
              f"  | TE n={m_te['n_trades']:3d} PF={m_te['profit_factor']:.2f} wr={m_te['win_rate']*100:.1f}% PnL={m_te['total_pnl']:+,.0f}  [{deg}]")

    # ======================== MICRO ACCOUNT ZAR 1000 ========================
    print("\n=== ZAR 1000 MICRO (max 5 pos, min 0.01 lot, 2000x) ===")
    acct1k = AccountSpec(starting_equity=1000, currency="ZAR", leverage=2000, fx_to_account={"USD":18.5})
    bt1k = BacktestConfig(starting_equity=1000, pay_entry_spread=True,
                          slippage_points_long=5, slippage_points_short=5,
                          max_open_positions=5, min_equity_fraction=0.30,
                          max_risk_per_trade=0.02, commission_per_lot_rt=0.0)
    for label, mask, cfg in [
        ("BASELINE", pd.Series(True,index=s.index), make_cfg()),
        ("LONG tp=0.6 r>=2.0", (s.dir_int==1)&(s.risk_atr>=2.0), make_cfg(tp1_r=0.6, min_risk=2.0)),
        ("LONG tp=0.75 r>=2.5", (s.dir_int==1)&(s.risk_atr>=2.5), make_cfg(tp1_r=0.75, min_risk=2.5)),
        ("LONG tp=0.5 r>=2.0", (s.dir_int==1)&(s.risk_atr>=2.0), make_cfg(tp1_r=0.5, min_risk=2.0)),
        ("LONG tp=0.6 r>=2.5", (s.dir_int==1)&(s.risk_atr>=2.5), make_cfg(tp1_r=0.6, min_risk=2.5)),
        ("LONG tp=0.6 r>=2.0 A+/A M15", (s.dir_int==1)&(s.risk_atr>=2.0)&s.grade.isin(["A+","A"])&(s.ob_tf=="M15"),
            make_cfg(tp1_r=0.6, min_risk=2.0, grades=("A+","A"), ob_tfs=("M15",))),
        ("LONG tp=0.75 r>=2.0 A+/A", (s.dir_int==1)&(s.risk_atr>=2.0)&s.grade.isin(["A+","A"]),
            make_cfg(tp1_r=0.75, min_risk=2.0, grades=("A+","A"))),
    ]:
        fsig = s[mask]
        sidx = build_sig_index(fsig, arr["time"])
        r = run_fast(arr, sidx, spec, acct1k, bt1k, cfg)
        m = calc_metrics(r.trades, equity=1000)
        print(f"  {fmt(m, label)}")

    # ======================== BEST CONFIGS — MONTH-BY-MONTH ROBUSTNESS ========================
    print("\n=== MONTH-BY-MONTH ROBUSTNESS: LONG tp=0.6 r>=2.0 ===")
    cfg_champ = make_cfg(tp1_r=0.6, min_risk=2.0)
    mask_champ = (s.dir_int==1)&(s.risk_atr>=2.0)
    months = sorted(s["month"].unique())
    good_months = 0; total_pnl = 0; worst_m = (None, 999)
    for m in months:
        m_mask = mask_champ & (s["month"]==m)
        fsig = s[m_mask]
        if len(fsig) < 2: continue
        sidx = build_sig_index(fsig, arr["time"])
        r = run_fast(arr, sidx, spec, acct, bt, cfg_champ)
        met = calc_metrics(r.trades)
        tag = "✓" if met["total_pnl"]>0 else "✗"
        if met["total_pnl"]>0: good_months += 1
        if met["total_pnl"] < worst_m[1]: worst_m = (str(m), met["total_pnl"])
        total_pnl += met["total_pnl"]
        print(f"  {str(m):10s} {tag} n={met['n_trades']:3d} PF={met['profit_factor']:.2f} PnL={met['total_pnl']:+,.0f} dd={met['max_drawdown_pct']:.1f}%")
    print(f"  -> {good_months}/{len(months)} profitable months, total PnL {total_pnl:+,.0f}")

    print("\n=== MONTH-BY-MONTH ROBUSTNESS: LONG tp=0.75 r>=2.5 ===")
    cfg_champ2 = make_cfg(tp1_r=0.75, min_risk=2.5)
    mask_champ2 = (s.dir_int==1)&(s.risk_atr>=2.5)
    good_months = 0; total_pnl = 0; worst_m = (None, 999)
    for m in months:
        m_mask = mask_champ2 & (s["month"]==m)
        fsig = s[m_mask]
        if len(fsig) < 2: continue
        sidx = build_sig_index(fsig, arr["time"])
        r = run_fast(arr, sidx, spec, acct, bt, cfg_champ2)
        met = calc_metrics(r.trades)
        tag = "✓" if met["total_pnl"]>0 else "✗"
        if met["total_pnl"]>0: good_months += 1
        if met["total_pnl"] < worst_m[1]: worst_m = (str(m), met["total_pnl"])
        total_pnl += met["total_pnl"]
        print(f"  {str(m):10s} {tag} n={met['n_trades']:3d} PF={met['profit_factor']:.2f} PnL={met['total_pnl']:+,.0f} dd={met['max_drawdown_pct']:.1f}%")
    print(f"  -> {good_months}/{len(months)} profitable months, total PnL {total_pnl:+,.0f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
