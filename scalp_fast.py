"""Vectorized fast backtester — ~2s per config vs 45s for engine.py.

Simulates positions/tranches exactly like engine.py but operates on numpy arrays
loaded once into memory. Same P&L math, same SL/TP/BE/trail logic.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from slytrade.config import DataConfig
from slytrade.strategy.config import StrategyConfig, ExitPlan, ConfluenceConfig, SessionFilter
from slytrade.backtest.specs import spec_for_symbol, AccountSpec, SymbolSpec
from slytrade.backtest.engine import BacktestConfig
from slytrade.backtest.positions import Position, Tranche, TrancheState, ExitReason


PRIORITY = [ExitReason.SL, ExitReason.BE, ExitReason.M15_CHOCH, ExitReason.M5_CHOCH,
            ExitReason.TIME_STOP, ExitReason.TRAIL, ExitReason.RUNNER_TARGET,
            ExitReason.TP2, ExitReason.TP1, ExitReason.END_OF_DATA]


def terminal_reason(reasons):
    for r in PRIORITY:
        if r in reasons: return r
    return reasons[-1] if reasons else ExitReason.END_OF_DATA


def zone_kind(sig):
    # Mirror engine._zone_kind: sig is a pandas Series (from iloc) or namedtuple
    ob = sig.ob_tf
    try:
        if ob is None:
            return "FVG"
        if pd.isna(ob):
            return "FVG"
    except (TypeError, ValueError):
        pass
    return "OB" if ob else "FVG"


@dataclass
class FastResult:
    trades: pd.DataFrame
    equity: np.ndarray


def run_fast(bars_arr, sig_by_idx, spec, acct, bt, scfg, starting_equity=20000.0):
    """Run backtest given:
       bars_arr = dict of numpy arrays (open, high, low, close, spread, atr14, m5_atr, m5_up, m5_dn, m15_up, m15_dn, time)
       sig_by_idx = dict {bar_idx: [sig_rec, ...]} where sig_rec has .direction .entry .stop .grade etc
    """
    n = bars_arr["n"]
    o = bars_arr["open"]; h = bars_arr["high"]; l = bars_arr["low"]; c = bars_arr["close"]
    sp = bars_arr["spread"]; atr1 = bars_arr["atr14"]; atr5 = bars_arr["m5_atr"]
    m5up = bars_arr["m5_up"]; m5dn = bars_arr["m5_dn"]; m15up = bars_arr["m15_up"]; m15dn = bars_arr["m15_dn"]
    t_arr = bars_arr["time"]
    point = spec.point; half_spread = point * 0.5 if bt.pay_entry_spread else 0.0
    slip_l = bt.slippage_points_long * point
    slip_s = bt.slippage_points_short * point
    exits = scfg.exits
    trail_mult = exits.runner_trail_atr_mult

    positions: List[Position] = []
    closed: List[Position] = []
    balance = starting_equity
    equity = starting_equity
    margin_used = 0.0
    pos_id = 0
    warmup = 500

    eq = np.full(n, starting_equity, dtype=np.float64)
    n_open = np.zeros(n, dtype=np.int32)

    for i in range(warmup, n):
        hi = h[i]; lo = l[i]; ci = c[i]; t = t_arr[i]
        sp_p = sp[i] * point if not np.isnan(sp[i]) else 0.0
        a1 = atr1[i] if not np.isnan(atr1[i]) and atr1[i]>0 else 10.0
        a5 = atr5[i] if not np.isnan(atr5[i]) and atr5[i]>0 else a1
        m5u = bool(m5up[i]); m5d = bool(m5dn[i])
        m15u = bool(m15up[i]); m15d = bool(m15dn[i])

        to_remove = []
        for pos in positions:
            pos.bars_held += 1
            d = pos.direction
            # MFE/MAE
            if d == 1:
                pos.max_favorable_excursion = max(pos.max_favorable_excursion, hi - pos.entry_price)
                pos.max_adverse_excursion = max(pos.max_adverse_excursion, pos.entry_price - lo)
            else:
                pos.max_favorable_excursion = max(pos.max_favorable_excursion, pos.entry_price - lo)
                pos.max_adverse_excursion = max(pos.max_adverse_excursion, hi - pos.entry_price)

            closed_flag = False

            # Emergency: M15 CHoCH
            if (d==1 and m15d) or (d==-1 and m15u):
                fill = lo - slip_s if d==1 else hi + slip_l
                for tr in pos.open_tranches():
                    tr.exit_price=fill; tr.exit_reason=ExitReason.M15_CHOCH; tr.exit_time=t
                    tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED
                pos.close_time=t; pos.close_reason=ExitReason.M15_CHOCH
                to_remove.append(pos); closed_flag=True

            if not closed_flag:
                # Update protective stops
                if pos.tp1_hit and not pos.be_lock:
                    for tr in pos.tranches:
                        if tr.state == TrancheState.OPEN:
                            tr.sl = pos.entry_price
                    pos.be_lock = True
                if pos.tp2_hit and pos.trail_sl is None:
                    pos.trail_sl = pos.tp1; pos.runner_trailing = True

                # Check exits per tranche
                for tr in list(pos.open_tranches()):
                    sl = tr.sl; tp = tr.tp
                    if d == 1:
                        sl_hit = (lo <= sl); tp_hit = (tp is not None and hi >= tp)
                        # Trail for T3
                        if pos.runner_trailing and tr.name=="T3" and pos.trail_sl is not None:
                            new_trail = max(pos.trail_sl, hi - trail_mult*a5)
                            if new_trail > pos.trail_sl: pos.trail_sl = new_trail
                            if lo <= pos.trail_sl and pos.trail_sl > pos.entry_price:
                                tr.exit_price = pos.trail_sl - slip_s
                                tr.exit_reason = ExitReason.TRAIL; tr.exit_time = t
                                tr.exit_bars = pos.bars_held; tr.state = TrancheState.CLOSED
                                continue
                        if sl_hit and tp_hit:
                            tr.exit_price=sl-slip_s; tr.exit_reason=ExitReason.SL
                            tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED; continue
                        if sl_hit:
                            reason = ExitReason.BE if (pos.be_lock and abs(sl-pos.entry_price)<1e-6) else ExitReason.SL
                            tr.exit_price=sl-slip_s; tr.exit_reason=reason
                            tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED; continue
                        if tp_hit:
                            fill = tp
                            if tr.name == "T1":
                                pos.tp1_hit = True
                                tr.exit_price=fill; tr.exit_reason=ExitReason.TP1
                                tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED
                                for tt in pos.tranches:
                                    if tt.state==TrancheState.OPEN: tt.sl=pos.entry_price
                                pos.be_lock=True; continue
                            elif tr.name == "T2":
                                pos.tp2_hit = True
                                tr.exit_price=fill; tr.exit_reason=ExitReason.TP2
                                tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED
                                for tt in pos.tranches:
                                    if tt.name=="T3" and tt.state==TrancheState.OPEN:
                                        tt.sl=pos.tp1; tt.tp=None
                                pos.trail_sl=pos.tp1; pos.runner_trailing=True; continue
                            elif tr.name == "T3" and tp is not None:
                                tr.exit_price=fill; tr.exit_reason=ExitReason.RUNNER_TARGET
                                tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED; continue
                    else:  # SHORT
                        sl_hit = (hi >= sl); tp_hit = (tp is not None and lo <= tp)
                        if pos.runner_trailing and tr.name=="T3" and pos.trail_sl is not None:
                            new_trail = min(pos.trail_sl, lo + trail_mult*a5)
                            if pos.trail_sl is None or new_trail < pos.trail_sl: pos.trail_sl = new_trail
                            if hi >= pos.trail_sl and pos.trail_sl < pos.entry_price:
                                tr.exit_price = pos.trail_sl + slip_l
                                tr.exit_reason=ExitReason.TRAIL; tr.exit_time=t
                                tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED; continue
                        if sl_hit and tp_hit:
                            tr.exit_price=sl+slip_l; tr.exit_reason=ExitReason.SL
                            tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED; continue
                        if sl_hit:
                            reason = ExitReason.BE if (pos.be_lock and abs(sl-pos.entry_price)<1e-6) else ExitReason.SL
                            tr.exit_price=sl+slip_l; tr.exit_reason=reason
                            tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED; continue
                        if tp_hit:
                            fill = tp
                            if tr.name == "T1":
                                pos.tp1_hit = True
                                tr.exit_price=fill; tr.exit_reason=ExitReason.TP1
                                tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED
                                for tt in pos.tranches:
                                    if tt.state==TrancheState.OPEN: tt.sl=pos.entry_price
                                pos.be_lock=True; continue
                            elif tr.name == "T2":
                                pos.tp2_hit = True
                                tr.exit_price=fill; tr.exit_reason=ExitReason.TP2
                                tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED
                                for tt in pos.tranches:
                                    if tt.name=="T3" and tt.state==TrancheState.OPEN:
                                        tt.sl=pos.tp1; tt.tp=None
                                pos.trail_sl=pos.tp1; pos.runner_trailing=True; continue
                            elif tr.name == "T3" and tp is not None:
                                tr.exit_price=fill; tr.exit_reason=ExitReason.RUNNER_TARGET
                                tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED; continue

                # Time stop
                if (not pos.is_closed()) and pos.bars_held >= exits.time_stop_bars:
                    # Compute current R total
                    r_per_lot = spec.profit_per_lot(pos.risk_per_unit_quote) * pos.total_lots
                    if r_per_lot > 0:
                        realized = pos.total_realized_pnl(spec)
                        unreal = pos.unrealized_pnl(ci, spec)
                        cur_r = (realized + unreal) / r_per_lot
                    else:
                        cur_r = 0
                    if cur_r < exits.time_stop_min_r:
                        fill = ci - slip_s if d==1 else ci + slip_l
                        for tr in pos.open_tranches():
                            tr.exit_price=fill; tr.exit_reason=ExitReason.TIME_STOP
                            tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED
                        pos.close_time=t; pos.close_reason=ExitReason.TIME_STOP
                        to_remove.append(pos); closed_flag=True

                # M5 CHoCH against T3 runner
                if not closed_flag:
                    t3 = next((tt for tt in pos.open_tranches() if tt.name=="T3"), None)
                    if t3 is not None and pos.runner_trailing:
                        against = (d==1 and m5d) or (d==-1 and m5u)
                        if against:
                            fill = lo - slip_s if d==1 else hi + slip_l
                            t3.exit_price=fill; t3.exit_reason=ExitReason.M5_CHOCH
                            t3.exit_time=t; t3.exit_bars=pos.bars_held; t3.state=TrancheState.CLOSED
                            if pos.is_closed():
                                pos.close_time=t; pos.close_reason=ExitReason.M5_CHOCH
                                to_remove.append(pos); closed_flag=True

                # Terminal reason if organic closure
                if not closed_flag and pos.is_closed() and pos.close_time is None:
                    last = [tr for tr in pos.tranches if tr.state==TrancheState.CLOSED and tr.exit_time==t]
                    term = terminal_reason([tr.exit_reason for tr in last]) if last else None
                    if term is None:
                        term = terminal_reason([tr.exit_reason for tr in pos.tranches if tr.exit_reason]) or ExitReason.END_OF_DATA
                    pos.close_time=t; pos.close_reason=term
                    to_remove.append(pos); closed_flag=True

        # Settle closed positions
        for pos in to_remove:
            if pos in positions:
                for tr in pos.tranches:
                    pnl = tr.realized_pnl_ccy(pos.direction, spec)
                    balance += acct.to_account_ccy(pnl, spec.currency_profit)
                margin_used -= (pos.total_lots * spec.contract_size * pos.entry_price)/max(bt.leverage,1)
                closed.append(pos); positions.remove(pos)

        # Open new positions
        for sig in sig_by_idx.get(i, []):
            if len(positions) >= bt.max_open_positions: continue
            if equity < bt.starting_equity * bt.min_equity_fraction: continue
            d = int(sig.direction)
            if d == 1:
                entry_fill = float(sig.entry) + (sp_p*0.5 if bt.pay_entry_spread else 0.0) + slip_l
            else:
                entry_fill = float(sig.entry) - (sp_p*0.5 if bt.pay_entry_spread else 0.0) - slip_s
            risk_per_unit = abs(entry_fill - float(sig.stop))
            if risk_per_unit <= 0: continue
            risk_pct = float(sig.risk_pct)
            risk_acct = risk_pct * equity
            risk_quote = risk_acct / acct.fx_to_account.get(spec.currency_profit, 1.0)
            lots = spec.lots_for_risk(risk_per_unit, risk_quote)
            if lots < spec.volume_min: continue
            actual_risk = spec.profit_per_lot(risk_per_unit) * lots
            actual_risk_acct = acct.to_account_ccy(actual_risk, spec.currency_profit)
            if actual_risk_acct > equity * bt.max_risk_per_trade: continue
            margin = (lots*spec.contract_size*entry_fill)/max(bt.leverage,1)
            margin_acct = acct.to_account_ccy(margin, spec.currency_profit)
            if margin_acct > equity*0.95: continue

            pos_id += 1
            tp1 = entry_fill + d * exits.tp1_r * risk_per_unit
            tp2 = entry_fill + d * exits.tp2_r * risk_per_unit
            tp_runner = float(getattr(sig, "tp_runner", entry_fill + d*3.0*risk_per_unit) or entry_fill + d*3.0*risk_per_unit)

            pos = Position(
                pos_id=pos_id, symbol=spec.name, direction=d,
                entry_time=t, entry_price=entry_fill, total_lots=lots,
                atr_at_entry=float(sig.atr_at_entry), grade=sig.grade,
                risk_pct=risk_pct, risk_per_unit_quote=risk_per_unit,
                initial_sl=float(sig.stop), tp1=float(tp1), tp2=float(tp2), tp_runner=float(tp_runner),
                swing_target_tf=getattr(sig, "swing_target_tf", "") or "",
                swing_target_price=float(getattr(sig, "swing_target_price", tp_runner) or tp_runner),
                trigger_tf=sig.trigger_tf, ob_tf=sig.ob_tf,
                zone_kind=zone_kind(sig),
                killzone=sig.killzone, session=sig.session,
                confluence_tags=list(sig.confluence) if hasattr(sig, "confluence") else [],
                htf_bias_summary=dict(getattr(sig, "htf_bias_summary", {}) or {}),
            )
            # Determine tranche fractions from exit plan
            t1f = exits.tp1_pct
            t2f = exits.tp2_pct
            t3f = max(0.0, 1.0 - t1f - t2f)
            pos.init_tranches(volume_min=spec.volume_min, volume_step=spec.volume_step,
                              t1_frac=t1f, t2_frac=t2f, t3_frac=t3f)
            # Deduct commission on entry
            comm = bt.commission_per_lot_rt * lots * 0.5
            balance -= acct.to_account_ccy(comm, spec.currency_profit)
            margin_used += margin
            positions.append(pos)

        # MTM
        mtm = sum(pos.unrealized_pnl(ci, spec) for pos in positions)
        equity = balance + acct.to_account_ccy(mtm, spec.currency_profit)
        eq[i] = equity
        n_open[i] = len(positions)

    # Close remaining at last close (no slip)
    ci = c[-1]; t = t_arr[-1]
    for pos in list(positions):
        d = pos.direction
        fill = ci
        for tr in pos.open_tranches():
            tr.exit_price=fill; tr.exit_reason=ExitReason.END_OF_DATA
            tr.exit_time=t; tr.exit_bars=pos.bars_held; tr.state=TrancheState.CLOSED
        pos.close_time=t; pos.close_reason=ExitReason.END_OF_DATA
        for tr in pos.tranches:
            pnl = tr.realized_pnl_ccy(pos.direction, spec)
            balance += acct.to_account_ccy(pnl, spec.currency_profit)
        closed.append(pos)
    positions = []
    eq[-1] = balance

    # Build trade df
    rows = []
    for pos in closed:
        pnl_q = pos.total_realized_pnl(spec)
        pnl_a = acct.to_account_ccy(pnl_q, spec.currency_profit)
        r = pos.r_multiple_realized(spec)
        rows.append(dict(pos_id=pos.pos_id, direction="LONG" if pos.direction==1 else "SHORT",
                         entry_time=pd.Timestamp(pos.entry_time, tz="UTC"), close_time=pd.Timestamp(pos.close_time, tz="UTC") if pos.close_time is not None else pd.NaT,
                         entry=pos.entry_price, initial_sl=pos.initial_sl, tp1=pos.tp1, tp2=pos.tp2,
                         grade=pos.grade, zone_kind=pos.zone_kind, ob_tf=pos.ob_tf or "",
                         trigger_tf=pos.trigger_tf, killzone=pos.killzone, session=pos.session,
                         atr_at_entry=pos.atr_at_entry, risk_pct=pos.risk_pct, lots=pos.total_lots,
                         bars_held=pos.bars_held, mfe=pos.max_favorable_excursion, mae=pos.max_adverse_excursion,
                         pnl_quote=pnl_q, pnl_acct=pnl_a, r_multiple=r, close_reason=pos.close_reason))
    return FastResult(trades=pd.DataFrame(rows), equity=eq)


def load_bars_and_signals():
    cfg = DataConfig.from_paths(Path("data/raw"), Path("data/processed"))
    need = ["time","open","high","low","close","spread","atr_14","M5_atr_14",
            "M5_major_choch_up","M5_major_choch_dn","M15_major_choch_up","M15_major_choch_dn"]
    base = cfg.aligned_path/"symbol=XAUUSDm"/"timeframe=M1"
    files = sorted(base.glob("**/*.parquet"))
    frames = []
    for f in files:
        pf = pq.ParquetFile(str(f))
        have = set(pf.schema.names)
        cols = [c for c in need if c in have]
        ch = pd.read_parquet(f, columns=cols)
        ch["time"] = pd.to_datetime(ch["time"], utc=True, errors="coerce")
        ch = ch.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)
        frames.append(ch)
    bars = pd.concat(frames, ignore_index=True)
    # Fill NaNs safely
    for col in ["spread","atr_14","M5_atr_14"]:
        if col in bars.columns:
            bars[col] = bars[col].ffill().bfill().fillna(0)
        else:
            bars[col] = 0.0
    for col in ["M5_major_choch_up","M5_major_choch_dn","M15_major_choch_up","M15_major_choch_dn"]:
        if col not in bars.columns:
            bars[col] = False
        else:
            bars[col] = bars[col].fillna(False).astype(bool)

    sdf = pd.read_parquet(cfg.signals_path/"symbol=XAUUSDm"/"signals.parquet")
    sdf["time"] = pd.to_datetime(sdf["time"], utc=True)
    return bars, sdf, cfg


def build_bars_arr(bars):
    tz = bars["time"].dt.tz
    # Convert to numpy datetime64 (UTC)
    t_ns = bars["time"].values.astype("datetime64[ns]")
    return dict(
        n=len(bars),
        open=bars["open"].to_numpy(dtype=np.float64),
        high=bars["high"].to_numpy(dtype=np.float64),
        low=bars["low"].to_numpy(dtype=np.float64),
        close=bars["close"].to_numpy(dtype=np.float64),
        spread=bars["spread"].to_numpy(dtype=np.float64),
        atr14=bars["atr_14"].to_numpy(dtype=np.float64),
        m5_atr=bars["M5_atr_14"].to_numpy(dtype=np.float64),
        m5_up=bars["M5_major_choch_up"].to_numpy(dtype=bool),
        m5_dn=bars["M5_major_choch_dn"].to_numpy(dtype=bool),
        m15_up=bars["M15_major_choch_up"].to_numpy(dtype=bool),
        m15_dn=bars["M15_major_choch_dn"].to_numpy(dtype=bool),
        time=t_ns,
    )


def build_sig_index(signals_df, bars_time_ns):
    """Map signal times to bar indices using a linear merge (both sorted)."""
    sig_times = pd.to_datetime(signals_df["time"], utc=True).values.astype("datetime64[ns]")
    # Sort signals by time
    order = np.argsort(sig_times, kind="mergesort")
    sig_times_sorted = sig_times[order]
    d = {}
    j = 0
    n_bars = len(bars_time_ns)
    for k in order:
        st = sig_times[k]
        while j < n_bars and bars_time_ns[j] < st:
            j += 1
        if j < n_bars and bars_time_ns[j] == st:
            d.setdefault(int(j), []).append(signals_df.iloc[k])
    return d


def calc_metrics(trades, equity=20000.0):
    if trades.empty:
        return dict(n_trades=0, win_rate=0, profit_factor=0, mean_R=0, total_pnl=0, max_drawdown_pct=0, return_pct=0, final_equity=equity)
    pnl = trades["pnl_acct"].astype(float)
    w = trades[pnl>0]; l = trades[pnl<0]; be = trades[pnl.abs()<1e-6]
    gw = float(w.pnl_acct.sum()) if len(w) else 0.0
    gl = float(-l.pnl_acct.sum()) if len(l) else 1e-9
    pf = gw / max(gl, 1e-9) if gl > 0 else float("inf")
    final_eq = equity + float(pnl.sum())
    running = equity + pnl.cumsum()
    rmax = running.cummax()
    dd = (running - rmax)/rmax
    return dict(n_trades=len(trades), n_win=len(w), n_loss=len(l), n_be=len(be),
                win_rate=float(len(w)/len(trades)), profit_factor=pf,
                mean_R=float(trades["r_multiple"].mean()), total_pnl=float(pnl.sum()),
                final_equity=float(final_eq), return_pct=float((final_eq-equity)/equity*100),
                max_drawdown_acct=float((running-rmax).min()),
                max_drawdown_pct=float(dd.min()*100))


def fmt(m, label=""):
    pf_str = "inf" if m["profit_factor"]==float("inf") else f"{m['profit_factor']:.2f}"
    return (f"{label:55s} n={m['n_trades']:4d} wr={m['win_rate']*100:5.1f}% PF={pf_str:>5s} "
            f"mR={m['mean_R']:+.3f} PnL={m['total_pnl']:+,.0f} ret={m['return_pct']:+.1f}% dd={m['max_drawdown_pct']:.1f}%")


if __name__ == "__main__":
    t0 = time.time()
    bars_df, sdf, _ = load_bars_and_signals()
    arr = build_bars_arr(bars_df)
    sig_idx_all = build_sig_index(sdf, arr["time"])
    spec = spec_for_symbol("XAUUSDm")
    acct = AccountSpec(starting_equity=20000, currency="ZAR", leverage=2000, fx_to_account={"USD":18.5})
    bt = BacktestConfig(starting_equity=20000, pay_entry_spread=True, slippage_points_long=5, slippage_points_short=5,
                        max_open_positions=10, min_equity_fraction=0.30, max_risk_per_trade=0.02,
                        commission_per_lot_rt=0.0)
    print(f"Loaded data in {time.time()-t0:.1f}s")

    def cfg(tp1_r=0.75, tp1_pct=1.0, tp2_r=1.5, tp2_pct=0.0, grades=("A+","A","B"),
            ob_tfs=("M15","M5"), zones=("OB",), min_risk=0.0, max_risk=99.0,
            atr_trail=0.5, time_stop=240, time_stop_min_r=0.0, be_buffer=0.05):
        return StrategyConfig(
            exits=ExitPlan(tp1_r=tp1_r, tp1_pct=tp1_pct, tp2_r=tp2_r, tp2_pct=tp2_pct,
                           ob_invalidation_buffer=be_buffer, runner_trail_atr_mult=atr_trail,
                           time_stop_bars=time_stop, time_stop_min_r=time_stop_min_r),
            confluence=ConfluenceConfig(accept_grades=grades, accept_ob_tfs=ob_tfs, accept_zone_kinds=zones,
                                        min_risk_atr=min_risk, max_risk_atr=max_risk),
            sessions=SessionFilter(block_off_hours=True, trade_asian_kz=False, trade_asian_range_retest=False))

    # Baseline
    t1=time.time()
    r = run_fast(arr, sig_idx_all, spec, acct, bt, cfg())
    print(fmt(calc_metrics(r.trades), "BASELINE") + f"  [{time.time()-t1:.2f}s]")

    # Pre-compute signal features for filtering
    s = sdf.copy()
    s["hour_utc"] = s["time"].dt.hour
    s["dow"] = s["time"].dt.dayofweek
    s["risk_atr"] = abs(s["entry"] - s["stop"]) / s["atr_at_entry"]
    s["dir_int"] = s["direction"].astype(int)

    def run_filtered(mask, scfg, label):
        fsig = s[mask]
        if len(fsig) < 5:
            print(f"{label:55s} SKIP n={len(fsig)}")
            return None
        sidx = build_sig_index(fsig, arr["time"])
        t1 = time.time()
        r = run_fast(arr, sidx, spec, acct, bt, scfg)
        m = calc_metrics(r.trades)
        star = " ***" if m["profit_factor"]>=1.8 and m["n_trades"]>=30 else ""
        print(f"{fmt(m, label)}{star}  [{time.time()-t1:.2f}s]")
        return m

    print("\n=== TP distance sweep (all signals) ===")
    for tp in [0.3,0.4,0.5,0.6,0.75,1.0,1.5]:
        run_filtered(pd.Series(True, index=s.index), cfg(tp1_r=tp), f"tp1={tp}R")

    print("\n=== Directional ===")
    run_filtered(s.dir_int==1, cfg(), "LONGS only")
    run_filtered(s.dir_int==-1, cfg(), "SHORTS only")

    print("\n=== Time-of-day ===")
    for (lo, hi, lbl) in [(7,9,"7-9h"),(8,9,"8-9h Lon open"),(8,10,"8-10h"),(8,14,"8-14h"),(12,15,"12-15h NY"),(13,14,"13-14h NY core"),(13,15,"13-15h")]:
        run_filtered(s.hour_utc.between(lo,hi), cfg(), lbl)

    print("\n=== Risk-in-ATR floor ===")
    for mr in [1.2,1.5,2.0,2.5,3.0,4.0]:
        run_filtered(s.risk_atr>=mr, cfg(min_risk=mr), f"risk>={mr} ATR")

    print("\n=== Grade filter ===")
    run_filtered(s.grade.isin(["A+","A"]), cfg(grades=("A+","A")), "A+/A only")
    run_filtered(s.grade=="A+", cfg(grades=("A+",)), "A+ only")
    run_filtered((s.grade=="A")|(s.grade=="A+"), cfg(grades=("A+","A"), tp1_r=0.5), "A+/A tp=0.5R")

    print("\n=== OB TF filter ===")
    run_filtered(s.ob_tf=="M15", cfg(ob_tfs=("M15",)), "M15 OB only")
    run_filtered(s.ob_tf=="M5", cfg(ob_tfs=("M5",)), "M5 OB only")

    print("\n=== COMBOS (hunt PF>1.8) ===")
    combos = [
        # LONGS focused
        ("LONG tp1=0.5R", s.dir_int==1, cfg(tp1_r=0.5)),
        ("LONG tp1=0.4R", s.dir_int==1, cfg(tp1_r=0.4)),
        ("LONG tp1=0.6R", s.dir_int==1, cfg(tp1_r=0.6)),
        ("LONG tp1=0.5R 8-14h", (s.dir_int==1)&s.hour_utc.between(8,14), cfg(tp1_r=0.5)),
        ("LONG tp1=0.5R 8-9h", (s.dir_int==1)&s.hour_utc.between(8,9), cfg(tp1_r=0.5)),
        ("LONG tp1=0.4R 8-14h", (s.dir_int==1)&s.hour_utc.between(8,14), cfg(tp1_r=0.4)),
        ("LONG tp1=0.5R risk>=2", (s.dir_int==1)&(s.risk_atr>=2.0), cfg(tp1_r=0.5, min_risk=2.0)),
        ("LONG tp1=0.5R risk>=3", (s.dir_int==1)&(s.risk_atr>=3.0), cfg(tp1_r=0.5, min_risk=3.0)),
        ("LONG tp1=0.5R 8-14h risk>=2", (s.dir_int==1)&s.hour_utc.between(8,14)&(s.risk_atr>=2.0), cfg(tp1_r=0.5, min_risk=2.0)),
        ("LONG tp1=0.4R 8-14h risk>=2", (s.dir_int==1)&s.hour_utc.between(8,14)&(s.risk_atr>=2.0), cfg(tp1_r=0.4, min_risk=2.0)),
        ("LONG tp1=0.5R M15", (s.dir_int==1)&(s.ob_tf=="M15"), cfg(tp1_r=0.5, ob_tfs=("M15",))),
        ("LONG tp1=0.5R A+/A", (s.dir_int==1)&s.grade.isin(["A+","A"]), cfg(tp1_r=0.5, grades=("A+","A"))),
        ("LONG tp1=0.5R 8-14h M15 A+/A", (s.dir_int==1)&s.hour_utc.between(8,14)&(s.ob_tf=="M15")&s.grade.isin(["A+","A"]), cfg(tp1_r=0.5, ob_tfs=("M15",), grades=("A+","A"))),
        ("LONG tp1=0.5R 8-14h risk>=2 A+/A", (s.dir_int==1)&s.hour_utc.between(8,14)&(s.risk_atr>=2.0)&s.grade.isin(["A+","A"]), cfg(tp1_r=0.5, min_risk=2.0, grades=("A+","A"))),
        # Both dirs
        ("tp1=0.4R all", pd.Series(True,index=s.index), cfg(tp1_r=0.4)),
        ("tp1=0.5R all", pd.Series(True,index=s.index), cfg(tp1_r=0.5)),
        ("tp1=0.5R risk>=2", s.risk_atr>=2.0, cfg(tp1_r=0.5, min_risk=2.0)),
        ("tp1=0.5R risk>=3", s.risk_atr>=3.0, cfg(tp1_r=0.5, min_risk=3.0)),
        ("tp1=0.4R risk>=2", s.risk_atr>=2.0, cfg(tp1_r=0.4, min_risk=2.0)),
        ("tp1=0.5R 8-9h", s.hour_utc.between(8,9), cfg(tp1_r=0.5)),
        ("tp1=0.4R 8-9h", s.hour_utc.between(8,9), cfg(tp1_r=0.4)),
        ("tp1=0.5R 8-14h", s.hour_utc.between(8,14), cfg(tp1_r=0.5)),
        ("tp1=0.5R M15 only", s.ob_tf=="M15", cfg(tp1_r=0.5, ob_tfs=("M15",))),
        ("tp1=0.5R 8-14h M15", s.hour_utc.between(8,14)&(s.ob_tf=="M15"), cfg(tp1_r=0.5, ob_tfs=("M15",))),
        # Laddered exits
        ("tp1=0.5R@50% tp2=1.0R@30% run20%", pd.Series(True,index=s.index), cfg(tp1_r=0.5, tp1_pct=0.5, tp2_r=1.0, tp2_pct=0.3)),
        ("LONG tp1=0.5@50% tp2=1.0@30%", s.dir_int==1, cfg(tp1_r=0.5, tp1_pct=0.5, tp2_r=1.0, tp2_pct=0.3)),
        # Tight time stop
        ("tp1=0.5R time_stop=120 (2h)", pd.Series(True,index=s.index), cfg(tp1_r=0.5, time_stop=120)),
        ("LONG tp1=0.5R time_stop=120", s.dir_int==1, cfg(tp1_r=0.5, time_stop=120)),
    ]
    for label, mask, scfg in combos:
        run_filtered(mask, scfg, label)

    print(f"\nTotal grid time: {time.time()-t0:.1f}s")
