"""Fast parameter sweep — limit to most promising combos."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import pandas as pd
from slytrade.backtest import run_backtest, BacktestConfig, AccountSpec
from slytrade.config import DataConfig
from slytrade.strategy.config import StrategyConfig, SetupGrades, ExitPlan, SessionFilter, ConfluenceConfig


def run_cfg(scfg, signals_df, equity=20000, slip=5):
    cfg = DataConfig.from_paths(Path("data/raw"), Path("data/processed"))
    acct = AccountSpec(starting_equity=equity, currency="ZAR", leverage=2000, fx_to_account={"USD":18.5})
    bt_cfg = BacktestConfig(starting_equity=equity, slippage_points_long=slip, slippage_points_short=slip,
                            max_open_positions=10, min_equity_fraction=0.30, max_risk_per_trade=0.02,
                            pay_entry_spread=True, commission_per_lot_rt=0.0)
    r = run_backtest(cfg, "XAUUSDm", signals_df, account=acct, bt_cfg=bt_cfg, strat_cfg=scfg,
                     progress=lambda m: None)
    return r.metrics


def fmt(m, label=""):
    if "error" in m: return f"{label:45s} ERROR: {m['error'][:40]}"
    return (f"{label:45s} n={m['n_trades']:4d} wr={m['win_rate']*100:5.1f}% PF={m['profit_factor']:.2f} "
            f"mR={m['mean_R']:+.3f} PnL={m['total_pnl']:+,.0f} dd={m.get('max_drawdown_pct',0):.1f}%")


def make_scfg(tp1_r=0.75, tp2_r=1.5, tp2_pct=0.0, grades=("A+","A","B"),
              ob_tfs=("M15","M5"), zones=("OB",), min_risk=1.2, max_risk=8.0,
              asian=False, asia_range=False, london_open=True, ny_open=True,
              london_kz=True, ny_kz=True, off_block=True, be_buffer=0.05,
              atr_trail=0.5):
    exits = ExitPlan(tp1_r=tp1_r, tp1_pct=1.0, tp2_r=tp2_r, tp2_pct=tp2_pct,
                     ob_invalidation_buffer=be_buffer, runner_trail_atr_mult=atr_trail)
    cc = ConfluenceConfig(
        accept_grades=grades, accept_ob_tfs=ob_tfs, accept_zone_kinds=zones,
        min_risk_atr=min_risk, max_risk_atr=max_risk,
    )
    sf = SessionFilter(
        trade_london_kz=london_kz, trade_ny_kz=ny_kz,
        trade_asian_kz=asian, trade_asian_range_retest=asia_range,
        trade_london_open30=london_open, trade_ny_open30=ny_open,
        block_off_hours=off_block,
    )
    return StrategyConfig(exits=exits, confluence=cc, sessions=sf)


def main():
    cfg = DataConfig.from_paths(Path("data/raw"), Path("data/processed"))
    sdf = pd.read_parquet(cfg.signals_path / "symbol=XAUUSDm/signals.parquet")
    sdf["time"] = pd.to_datetime(sdf["time"], utc=True)
    s = sdf.copy()
    s["hour_utc"] = s["time"].dt.hour
    s["dow"] = s["time"].dt.dayofweek
    s["risk_atr"] = s["risk_per_unit"] / s["atr_at_entry"]

    print(fmt(run_cfg(make_scfg(), sdf), "BASELINE"))

    # Quick single-parameter changes first
    print("\n== TP1 distance ==")
    for tp in [0.4, 0.5, 0.6, 0.75, 1.0]:
        print(fmt(run_cfg(make_scfg(tp1_r=tp), sdf), f"tp1={tp}R"))

    print("\n== Time-of-day filters ==")
    for hlo, hhi, label in [(8,9,"8-9 (Lon open)"), (8,10,"8-10"), (12,14,"12-14 NY"), (13,14,"13-14 NY core"), (7,10,"7-10")]:
        m = (s["hour_utc"]>=hlo)&(s["hour_utc"]<=hhi)
        print(fmt(run_cfg(make_scfg(), s[m]), label))

    print("\n== Direction ==")
    print(fmt(run_cfg(make_scfg(), s[s.direction==1]), "LONGS only"))
    print(fmt(run_cfg(make_scfg(), s[s.direction==-1]), "SHORTS only"))

    print("\n== Risk/ATR floor ==")
    for mr in [1.2, 2.0, 2.5, 3.0]:
        print(fmt(run_cfg(make_scfg(min_risk=mr), sdf), f"risk>={mr} ATR"))

    print("\n== Grades ==")
    for g, label in [(("A+","A"),"A+/A"),(("A+",),"A+ only")]:
        print(fmt(run_cfg(make_scfg(grades=g), sdf), label))

    print("\n== OB TFs ==")
    for tf, label in [(("M15",),"M15 only"),(("M5",),"M5 only")]:
        print(fmt(run_cfg(make_scfg(ob_tfs=tf), sdf), label))

    print("\n== COMBINATIONS (most promising) ==")
    combos = [
        ("LONGS only", dict(grades=("A+","A","B"), ob_tfs=("M15","M5")), s.direction==1),
        ("LONGS + tp1=0.5R", dict(tp1_r=0.5), s.direction==1),
        ("LONGS + 8-14h", dict(), (s.direction==1) & s.hour_utc.between(8,14)),
        ("LONGS + risk>=2.0", dict(min_risk=2.0), (s.direction==1) & (s.risk_atr>=2.0)),
        ("LONGS + tp1=0.5R + risk>=2.0", dict(tp1_r=0.5, min_risk=2.0), (s.direction==1)&(s.risk_atr>=2.0)),
        ("LONGS + tp1=0.5R + 8-14h", dict(tp1_r=0.5), (s.direction==1)&s.hour_utc.between(8,14)),
        ("tp1=0.5R + risk>=2.0", dict(tp1_r=0.5, min_risk=2.0), s.risk_atr>=2.0),
        ("tp1=0.5R + risk>=3.0", dict(tp1_r=0.5, min_risk=3.0), s.risk_atr>=3.0),
        ("LONGS + tp1=0.5R + risk>=3.0", dict(tp1_r=0.5, min_risk=3.0), (s.direction==1)&(s.risk_atr>=3.0)),
        ("8-9h London open", dict(), s.hour_utc.between(8,9)),
        ("8-9h + tp1=0.5R", dict(tp1_r=0.5), s.hour_utc.between(8,9)),
        ("8-9h LONGS", dict(), (s.direction==1)&s.hour_utc.between(8,9)),
        ("8-9h SHORTS", dict(), (s.direction==-1)&s.hour_utc.between(8,9)),
        ("M15 OBs only", dict(ob_tfs=("M15",)), s.ob_tf=="M15"),
        ("M15 + LONGS", dict(ob_tfs=("M15",)), (s.direction==1)&(s.ob_tf=="M15")),
        ("A+/A only", dict(grades=("A+","A")), s.grade.isin(["A+","A"])),
        ("A+/A + LONGS", dict(grades=("A+","A")), (s.direction==1)&s.grade.isin(["A+","A"])),
        ("A+/A + tp1=0.5R", dict(grades=("A+","A"), tp1_r=0.5), s.grade.isin(["A+","A"])),
        ("A+/A + risk>=2.0", dict(grades=("A+","A"), min_risk=2.0), s.grade.isin(["A+","A"])&(s.risk_atr>=2.0)),
        ("A+/A + LONGS + tp1=0.5R", dict(grades=("A+","A"), tp1_r=0.5), (s.direction==1)&s.grade.isin(["A+","A"])),
        ("LONGS + M15 + A+/A", dict(grades=("A+","A"), ob_tfs=("M15",)), (s.direction==1)&(s.ob_tf=="M15")&s.grade.isin(["A+","A"])),
        ("LONGS + M15 + tp1=0.5R", dict(tp1_r=0.5, ob_tfs=("M15","M5")), (s.direction==1)),
        ("LONGS + risk>=2 + M15 + A+/A", dict(grades=("A+","A"), ob_tfs=("M15",), min_risk=2.0), (s.direction==1)&(s.ob_tf=="M15")&s.grade.isin(["A+","A"])&(s.risk_atr>=2.0)),
        ("LONGS + 8-14h + tp1=0.5R", dict(tp1_r=0.5), (s.direction==1)&s.hour_utc.between(8,14)),
        ("LONGS + 8-14h + tp1=0.5R + risk>=2", dict(tp1_r=0.5, min_risk=2.0), (s.direction==1)&s.hour_utc.between(8,14)&(s.risk_atr>=2)),
    ]
    for label, kw, mask in combos:
        f = s[mask].copy()
        if len(f) < 10:
            continue
        scfg = make_scfg(**kw)
        m = run_cfg(scfg, f)
        print(fmt(m, label))


if __name__ == "__main__":
    main()
