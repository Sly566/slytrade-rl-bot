"""Layer 5 — streaming hedging backtest engine (v0.8.1 scalp-tuned).

Walks aligned M1 bars in time order. At every bar:
  1. Update open positions with this bar's OHLC → check SL/TP/CHoCH/time-stop
     triggers, update ATR trailing stops, close tranches as needed.
  2. If any signal fires at THIS bar's open (strictly causal: signal time =
     bar open time), open a new hedged position sized per grade.

Hedging mode: a new setup in the same direction as an existing open adds
a second (third, ...) independent position. A setup against an open
position does NOT close the existing position — that's handled by the
existing SL/CHoCH/trail logic on the incumbent.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import DataConfig
from ..strategy.config import StrategyConfig
from .positions import (
    ExitReason,
    Position,
    Tranche,
    TrancheState,
)
from .specs import AccountSpec, SymbolSpec, spec_for_symbol

ProgressFn = Callable[[str], None]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    tranches: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict
    n_bars: int = 0
    n_signals: int = 0

    def summary(self) -> str:
        import json
        return json.dumps(self.metrics, indent=2, default=str)


@dataclass
class BacktestConfig:
    starting_equity: float = 2000.0
    account_ccy: str = "ZAR"
    leverage: int = 2000
    usd_zar: float = 18.5
    slippage_points_long: int = 5
    slippage_points_short: int = 5
    commission_per_lot_rt: float = 0.0
    pay_entry_spread: bool = True
    max_open_positions: int = 10
    min_equity_fraction: float = 0.30
    max_risk_per_trade: float = 0.02


class BacktestEngine:
    def __init__(
        self,
        symbol_spec: SymbolSpec,
        account: AccountSpec,
        cfg: StrategyConfig | None = None,
        bt_cfg: BacktestConfig | None = None,
        progress: ProgressFn | None = None,
    ):
        self.spec = symbol_spec
        self.acct = account
        self.cfg = cfg or StrategyConfig()
        self.bt = bt_cfg or BacktestConfig()
        self.progress = progress or (lambda _m: None)

        self._pos_id = 0
        self.positions: list[Position] = []
        self.closed: list[Position] = []

        self.equity = float(self.bt.starting_equity)
        self.balance = float(self.bt.starting_equity)
        self.margin_used_quote = 0.0
        self._eq_rows: list[dict] = []

    @staticmethod
    def _zone_kind(sig) -> str:
        ob = getattr(sig, "ob_tf", None)
        if ob is None:
            return "FVG"
        try:
            if pd.isna(ob):
                return "FVG"
        except (TypeError, ValueError):
            pass
        return "OB" if ob else "FVG"

    _REASON_PRIORITY = [
        ExitReason.SL, ExitReason.BE, ExitReason.M15_CHOCH, ExitReason.M5_CHOCH,
        ExitReason.TIME_STOP, ExitReason.TRAIL, ExitReason.RUNNER_TARGET,
        ExitReason.TP2, ExitReason.TP1, ExitReason.END_OF_DATA,
    ]

    @classmethod
    def _terminal_reason(cls, reasons):
        for r in cls._REASON_PRIORITY:
            if r in reasons:
                return r
        return reasons[-1] if reasons else None

    def _open_position(self, row: pd.Series, sig) -> Position | None:
        if len(self.positions) >= self.bt.max_open_positions:
            return None
        if self.equity < self.bt.starting_equity * self.bt.min_equity_fraction:
            return None
        d = int(sig.direction)
        # Directional / persona filters.
        # When persona_gating is True (default paper-trade persona, v0.9.0), we
        # hard-reject signals that don't match the profitable long-only config.
        # When persona_gating is False (RL training mode), we trust the signal
        # generator to have emitted the right candidates and open every signal
        # that reaches the engine — the agent decides to act/skip.
        if self.cfg.confluence.persona_gating:
            # Directional filter (Layer 5: shorts PF 1.01 -> disabled by default)
            if d == 1 and not self.cfg.confluence.accept_longs:
                return None
            if d == -1 and not self.cfg.confluence.accept_shorts:
                return None
            # Risk-in-ATR floor filter (Layer 5: tight stops get hunted)
            _atr = float(sig.atr_at_entry) if float(sig.atr_at_entry) > 0 else 1.0
            _risk_w = abs(float(sig.entry) - float(sig.stop))
            _risk_atr = _risk_w / _atr if _atr > 0 else 999.0
            if _risk_atr < self.cfg.confluence.min_risk_atr:
                return None
            if _risk_atr > self.cfg.confluence.max_risk_atr:
                return None
            # OB TF / grade filters
            obtf = getattr(sig, "ob_tf", None)
            try:
                if obtf is None or (isinstance(obtf, float) and pd.isna(obtf)):
                    obtf = None
            except (TypeError, ValueError):
                pass
            if obtf is not None and obtf not in self.cfg.confluence.accept_ob_tfs:
                return None
            if str(sig.grade) not in self.cfg.confluence.accept_grades:
                return None
        # Always compute these for fill math:
        _atr = float(sig.atr_at_entry) if float(sig.atr_at_entry) > 0 else 1.0
        _risk_w = abs(float(sig.entry) - float(sig.stop))
        spread_pts = float(row.get("spread", 0.0))
        half_spread_price = spread_pts * self.spec.point * 0.5 if self.bt.pay_entry_spread else 0.0
        slip_price_long = self.bt.slippage_points_long * self.spec.point
        slip_price_short = self.bt.slippage_points_short * self.spec.point
        if int(sig.direction) == 1:
            fill = float(sig.entry) + half_spread_price + slip_price_long
        else:
            fill = float(sig.entry) - half_spread_price - slip_price_short
        risk_quote_per_unit = abs(fill - float(sig.stop))
        if risk_quote_per_unit <= 0:
            return None
        risk_acct_ccy = float(sig.risk_pct) * self.equity
        risk_quote_ccy = risk_acct_ccy / self.acct.fx_to_account.get(self.spec.currency_profit, 1.0)
        lots = self.spec.lots_for_risk(risk_quote_per_unit, risk_quote_ccy)
        if lots < self.spec.volume_min:
            return None
        actual_risk_quote = self.spec.profit_per_lot(risk_quote_per_unit) * lots
        actual_risk_acct = self.acct.to_account_ccy(actual_risk_quote, self.spec.currency_profit)
        if actual_risk_acct > self.equity * self.bt.max_risk_per_trade:
            return None
        margin_quote = (lots * self.spec.contract_size * fill) / max(self.bt.leverage, 1)
        margin_acct = self.acct.to_account_ccy(margin_quote, self.spec.currency_profit)
        if margin_acct > self.equity * 0.95:
            return None
        self._pos_id += 1

        # Compute TPs based on cfg.exits
        exits = self.cfg.exits
        tp1 = fill + sig.direction * exits.tp1_r * risk_quote_per_unit
        tp2 = fill + sig.direction * exits.tp2_r * risk_quote_per_unit
        tp_runner = float(getattr(sig, "tp_runner", fill + sig.direction * 3.0 * risk_quote_per_unit)
                          or fill + sig.direction * 3.0 * risk_quote_per_unit)

        pos = Position(
            pos_id=self._pos_id, symbol=self.spec.name, direction=int(sig.direction),
            entry_time=row["time"], entry_price=fill, total_lots=lots,
            atr_at_entry=float(sig.atr_at_entry), grade=sig.grade,
            risk_pct=float(sig.risk_pct), risk_per_unit_quote=risk_quote_per_unit,
            initial_sl=float(sig.stop),
            tp1=float(tp1), tp2=float(tp2), tp_runner=float(tp_runner),
            swing_target_tf=getattr(sig, "swing_target_tf", "") or "",
            swing_target_price=float(getattr(sig, "swing_target_price", tp_runner) or tp_runner),
            trigger_tf=sig.trigger_tf, ob_tf=sig.ob_tf,
            zone_kind=self._zone_kind(sig),
            killzone=sig.killzone, session=sig.session,
            confluence_tags=list(sig.confluence) if hasattr(sig, "confluence") else [],
            htf_bias_summary=dict(getattr(sig, "htf_bias_summary", {}) or {}),
        )
        pos.init_tranches(volume_min=self.spec.volume_min, volume_step=self.spec.volume_step,
                          t1_frac=exits.tp1_pct,
                          t2_frac=exits.tp2_pct if exits.tp2_pct > 0 else 0.0,
                          t3_frac=max(0.0, 1.0 - exits.tp1_pct - exits.tp2_pct))
        entry_cost_quote = self.bt.commission_per_lot_rt * lots * 0.5
        entry_cost_acct = self.acct.to_account_ccy(entry_cost_quote, self.spec.currency_profit)
        self.balance -= entry_cost_acct
        self.equity -= entry_cost_acct
        self.margin_used_quote += margin_quote
        self.positions.append(pos)
        return pos

    def _process_bar(self, row: pd.Series, signals_this_bar: list) -> None:
        t = row["time"]
        h = float(row["high"]); l = float(row["low"])
        float(row["open"]); c = float(row["close"])
        atr_m1 = float(row.get("atr_14", np.nan))
        atr_m5 = float(row.get("M5_atr_14", atr_m1))
        spread_pts = float(row.get("spread", 0))
        half_spread = spread_pts * self.spec.point * 0.5 if self.bt.pay_entry_spread else 0.0
        slip_l = self.bt.slippage_points_long * self.spec.point
        slip_s = self.bt.slippage_points_short * self.spec.point
        m5_up = bool(row.get("M5_major_choch_up", False))
        m5_dn = bool(row.get("M5_major_choch_dn", False))
        m15_up = bool(row.get("M15_major_choch_up", False))
        m15_dn = bool(row.get("M15_major_choch_dn", False))

        positions_to_remove = []
        for pos in self.positions:
            pos.bars_held += 1
            pos.update_excursion(h, l)
            d = pos.direction
            live_atr = atr_m5 if pos.runner_trailing else (atr_m1 if not np.isnan(atr_m1) and atr_m1 > 0 else pos.atr_at_entry)
            if np.isnan(live_atr) or live_atr <= 0:
                live_atr = pos.atr_at_entry
            closed_this_bar = False

            # Emergency exit: M15 CHoCH against -> close all immediately
            emergency = (d == 1 and m15_dn) or (d == -1 and m15_up)
            if emergency:
                fill = l - slip_s if d == 1 else h + slip_l
                self._close_position(pos, fill, ExitReason.M15_CHOCH, t)
                positions_to_remove.append(pos); closed_this_bar = True

            if not closed_this_bar:
                self._update_protective_stops(pos, live_atr)
                self._check_tranche_exits(pos, h, l, t, half_spread, slip_l, slip_s,
                                          m5_up, m5_dn, atr_m5)
                # Time stop
                if (not pos.is_closed() and pos.bars_held >= self.cfg.exits.time_stop_bars):
                    cur_r = pos.r_multiple_total(c, self.spec)
                    if cur_r < self.cfg.exits.time_stop_min_r:
                        fill = c - slip_s if d == 1 else c + slip_l
                        self._close_position(pos, fill, ExitReason.TIME_STOP, t)
                        positions_to_remove.append(pos); closed_this_bar = True
                # M5 CHoCH runner exit
                if not closed_this_bar:
                    t3 = next((tt for tt in pos.open_tranches() if tt.name == "T3"), None)
                    if t3 is not None and pos.runner_trailing:
                        m5_against = (d == 1 and m5_dn) or (d == -1 and m5_up)
                        if m5_against:
                            fill = l - slip_s if d == 1 else h + slip_l
                            pos.close_tranche("T3", fill, ExitReason.M5_CHOCH, t)
                            self._realize(pos, t3)
                            if pos.is_closed():
                                pos.close_time = t; pos.close_reason = ExitReason.M5_CHOCH
                                positions_to_remove.append(pos); closed_this_bar = True
                # Terminal reason propagation for organic tranche closures
                if not closed_this_bar and pos.is_closed() and pos.close_time is None:
                    last = [tr for tr in pos.tranches
                            if tr.state == TrancheState.CLOSED and tr.exit_time == t]
                    terminal = self._terminal_reason([tr.exit_reason for tr in last]) if last else None
                    if terminal is None:
                        terminal = self._terminal_reason(
                            [tr.exit_reason for tr in pos.tranches if tr.exit_reason]
                        ) or ExitReason.END_OF_DATA
                    pos.close_time = t; pos.close_reason = terminal
                    positions_to_remove.append(pos); closed_this_bar = True

        for pos in positions_to_remove:
            if pos in self.positions:
                margin_quote = (pos.total_lots * self.spec.contract_size * pos.entry_price) / max(self.bt.leverage, 1)
                self.margin_used_quote -= margin_quote
                self.closed.append(pos); self.positions.remove(pos)

        for sig in signals_this_bar:
            self._open_position(row, sig)

        mtm = 0.0
        for pos in self.positions:
            mtm += pos.unrealized_pnl(c, self.spec)
        mtm_acct = self.acct.to_account_ccy(mtm, self.spec.currency_profit)
        self.equity = self.balance + mtm_acct
        self._eq_rows.append({
            "time": t, "equity": self.equity, "balance": self.balance,
            "n_open": len(self.positions),
            "open_lots": sum(p.total_lots for p in self.positions),
            "margin_used_acct": self.acct.to_account_ccy(self.margin_used_quote, self.spec.currency_profit),
        })

    def _update_protective_stops(self, pos: Position, atr: float) -> None:
        """Move stops based on exit plan state."""
        # After TP1: move remaining tranches' SL to break-even
        if pos.tp1_hit and not pos.be_lock:
            for t in pos.tranches:
                if t.state == TrancheState.OPEN:
                    t.sl = pos.entry_price
            pos.be_lock = True
        # After TP2: start trailing the runner (T3)
        if pos.tp2_hit and pos.trail_sl is None:
            pos.trail_sl = pos.tp1; pos.runner_trailing = True

    def _check_tranche_exits(self, pos, h, l, t, half_spread, slip_l, slip_s,
                             m5_up, m5_dn, atr_m5):
        exits = self.cfg.exits
        d = pos.direction
        for tr in list(pos.open_tranches()):
            sl = tr.sl; tp = tr.tp
            if d == 1:  # LONG
                sl_hit = (l <= sl)
                tp_hit = (tp is not None and h >= tp)
                # Update trailing stop for T3 runner
                if pos.runner_trailing and tr.name == "T3" and pos.trail_sl is not None:
                    trail_dist = exits.runner_trail_atr_mult * atr_m5
                    new_trail = max(pos.trail_sl, h - trail_dist)
                    if new_trail > pos.trail_sl:
                        pos.trail_sl = new_trail
                    if l <= pos.trail_sl and pos.trail_sl > pos.entry_price:
                        fill = pos.trail_sl - slip_s
                        pos.close_tranche(tr.name, fill, ExitReason.TRAIL, t); self._realize(pos, tr); continue
                if sl_hit and tp_hit:
                    # Can't tell which came first on M1 — be conservative, assume SL
                    fill = sl - slip_s
                    pos.close_tranche(tr.name, fill, ExitReason.SL, t); self._realize(pos, tr); continue
                if sl_hit:
                    fill = sl - slip_s
                    reason = ExitReason.BE if (pos.be_lock and abs(sl - pos.entry_price) < 1e-6) else ExitReason.SL
                    pos.close_tranche(tr.name, fill, reason, t); self._realize(pos, tr); continue
                if tp_hit:
                    fill = tp
                    if tr.name == "T1":
                        pos.tp1_hit = True
                        pos.close_tranche("T1", fill, ExitReason.TP1, t); self._realize(pos, tr)
                        # Move remaining open tranches to BE after TP1
                        for tt in pos.tranches:
                            if tt.state == TrancheState.OPEN:
                                tt.sl = pos.entry_price
                        pos.be_lock = True; continue
                    elif tr.name == "T2":
                        pos.tp2_hit = True
                        pos.close_tranche("T2", fill, ExitReason.TP2, t); self._realize(pos, tr)
                        for tt in pos.tranches:
                            if tt.name == "T3" and tt.state == TrancheState.OPEN:
                                tt.sl = pos.tp1; tt.tp = None  # T3 becomes runner
                        pos.trail_sl = pos.tp1; pos.runner_trailing = True; continue
                    elif tr.name == "T3" and tp is not None:
                        pos.close_tranche("T3", fill, ExitReason.RUNNER_TARGET, t); self._realize(pos, tr); continue
            else:  # SHORT
                if pos.runner_trailing and tr.name == "T3" and pos.trail_sl is not None:
                    trail_dist = exits.runner_trail_atr_mult * atr_m5
                    new_trail = min(pos.trail_sl, l + trail_dist)
                    if pos.trail_sl is None or new_trail < pos.trail_sl:
                        pos.trail_sl = new_trail
                    sl_hit = (h >= pos.trail_sl)
                    if h >= pos.trail_sl and pos.trail_sl < pos.entry_price:
                        fill = pos.trail_sl + slip_l
                        pos.close_tranche(tr.name, fill, ExitReason.TRAIL, t); self._realize(pos, tr); continue
                sl_hit = (h >= sl)
                tp_hit = (tp is not None and l <= tp)
                if sl_hit and tp_hit:
                    fill = sl + slip_l
                    pos.close_tranche(tr.name, fill, ExitReason.SL, t); self._realize(pos, tr); continue
                if sl_hit:
                    fill = sl + slip_l
                    reason = ExitReason.BE if (pos.be_lock and abs(sl - pos.entry_price) < 1e-6) else ExitReason.SL
                    pos.close_tranche(tr.name, fill, reason, t); self._realize(pos, tr); continue
                if tp_hit:
                    fill = tp
                    if tr.name == "T1":
                        pos.tp1_hit = True
                        pos.close_tranche("T1", fill, ExitReason.TP1, t); self._realize(pos, tr)
                        for tt in pos.tranches:
                            if tt.state == TrancheState.OPEN:
                                tt.sl = pos.entry_price
                        pos.be_lock = True; continue
                    elif tr.name == "T2":
                        pos.tp2_hit = True
                        pos.close_tranche("T2", fill, ExitReason.TP2, t); self._realize(pos, tr)
                        for tt in pos.tranches:
                            if tt.name == "T3" and tt.state == TrancheState.OPEN:
                                tt.sl = pos.tp1; tt.tp = None
                        pos.trail_sl = pos.tp1; pos.runner_trailing = True; continue
                    elif tr.name == "T3" and tp is not None:
                        pos.close_tranche("T3", fill, ExitReason.RUNNER_TARGET, t); self._realize(pos, tr); continue

    def _realize(self, pos: Position, tranche: Tranche) -> None:
        pnl_quote = tranche.realized_pnl_ccy(pos.direction, self.spec)
        comm_quote = self.bt.commission_per_lot_rt * tranche.lots * 0.5
        pnl_quote -= comm_quote
        pnl_acct = self.acct.to_account_ccy(pnl_quote, self.spec.currency_profit)
        self.balance += pnl_acct

    def _close_position(self, pos: Position, price: float, reason: str, t, slip=0) -> None:
        for tr in pos.open_tranches():
            tr.exit_price = price; tr.exit_reason = reason
            tr.exit_time = t; tr.exit_bars = pos.bars_held; tr.state = TrancheState.CLOSED
            self._realize(pos, tr)
        pos.close_time = t; pos.close_reason = reason

    def run(self, m1_files: list[Path], signals_df: pd.DataFrame) -> BacktestResult:
        signals_df = signals_df.copy()
        signals_df["time"] = pd.to_datetime(signals_df["time"], utc=True)
        # Index signals by bar open time
        sig_by_time: dict = {}
        for rec in signals_df.itertuples(index=False):
            sig_by_time.setdefault(rec.time, []).append(rec)
        self.progress(f"Backtesting {len(signals_df):,} signals over {len(m1_files)} partitions...")
        total_rows = 0; n_signals = 0; warmup = True; warmup_rows = 1000
        for _f_idx, f in enumerate(m1_files):
            try:
                need = ["time","open","high","low","close","spread","tick_volume",
                        "atr_14","M5_atr_14",
                        "M5_major_choch_up","M5_major_choch_dn",
                        "M15_major_choch_up","M15_major_choch_dn"]
                pf = pq.ParquetFile(str(f)); have = set(pf.schema.names)
                cols = [c for c in need if c in have]
                chunk = pd.read_parquet(f, columns=cols)
            except Exception as e:
                self.progress(f"  skip {f.name}: {e}"); continue
            chunk["time"] = pd.to_datetime(chunk["time"], utc=True)
            chunk = chunk.sort_values("time", kind="mergesort").reset_index(drop=True)
            for _i, row in chunk.iterrows():
                if warmup:
                    warmup_rows -= 1
                    if warmup_rows <= 0: warmup = False
                    continue
                sigs_here = sig_by_time.get(row["time"], [])
                n_signals += len(sigs_here)
                self._process_bar(row, sigs_here)
                total_rows += 1
            self.progress(f"  {f.parent.name}/{f.name}: {len(chunk):,} rows, "
                          f"closed={len(self.closed)}, open={len(self.positions)}, "
                          f"equity={self.equity:.2f}")
        if self.positions:
            # Capture last close from the final equity row by re-scanning the last file
            # (simpler: mark all open to market at the last bar's close, stored as part of eq_rows via a tracked "last_close")
            last_t = None
            if self._eq_rows:
                last_t = self._eq_rows[-1]["time"]
            # We don't have last close tracked; use entry (conservative zero-PnL at EOD)
            for pos in list(self.positions):
                fill = pos.entry_price
                self._close_position(pos, fill, ExitReason.END_OF_DATA, last_t)
                margin_quote = (pos.total_lots * self.spec.contract_size * pos.entry_price) / max(self.bt.leverage,1)
                self.margin_used_quote -= margin_quote
                self.closed.append(pos)
            self.positions = []

        trade_rows = []; tranche_rows = []
        for pos in self.closed:
            r_total = pos.r_multiple_realized(self.spec)
            pnl_quote = pos.total_realized_pnl(self.spec)
            pnl_acct = self.acct.to_account_ccy(pnl_quote, self.spec.currency_profit)
            trade_rows.append({
                "pos_id": pos.pos_id, "symbol": pos.symbol,
                "direction": "LONG" if pos.direction == 1 else "SHORT",
                "entry_time": pos.entry_time, "close_time": pos.close_time,
                "entry": pos.entry_price, "initial_sl": pos.initial_sl,
                "tp1": pos.tp1, "tp2": pos.tp2, "tp_runner": pos.tp_runner,
                "grade": pos.grade, "zone_kind": pos.zone_kind,
                "ob_tf": pos.ob_tf or "", "trigger_tf": pos.trigger_tf,
                "session": pos.session, "killzone": pos.killzone,
                "atr_at_entry": pos.atr_at_entry,
                "risk_pct": pos.risk_pct, "lots": pos.total_lots,
                "bars_held": pos.bars_held,
                "mfe": pos.max_favorable_excursion, "mae": pos.max_adverse_excursion,
                "pnl_quote": pnl_quote, "pnl_acct": pnl_acct,
                "r_multiple": r_total, "close_reason": pos.close_reason,
            })
            for t in pos.tranches:
                pnl_t = t.realized_pnl_ccy(pos.direction, self.spec)
                tranche_rows.append({
                    "pos_id": pos.pos_id, "tranche": t.name,
                    "size_frac": t.size_frac, "lots": t.lots,
                    "entry": t.entry, "sl": t.sl, "tp": t.tp,
                    "exit_price": t.exit_price, "exit_reason": t.exit_reason,
                    "exit_bars": t.exit_bars, "pnl_quote": pnl_t,
                })
        trades_df = pd.DataFrame(trade_rows); tranches_df = pd.DataFrame(tranche_rows)
        eq_df = pd.DataFrame(self._eq_rows)
        metrics = self._compute_metrics(trades_df, tranches_df, eq_df)
        return BacktestResult(
            trades=trades_df, tranches=tranches_df, equity_curve=eq_df,
            metrics=metrics, n_bars=total_rows, n_signals=n_signals,
        )

    def _compute_metrics(self, trades, tranches, eq) -> dict:
        m: dict = {}
        if trades.empty:
            m["error"] = "no trades"; return m
        pnl = trades["pnl_acct"].astype(float)
        wins = trades[pnl > 0]; losses = trades[pnl < 0]; bes = trades[pnl.abs() < 1e-6]
        m["n_trades"] = int(len(trades)); m["n_win"] = int(len(wins)); m["n_loss"] = int(len(losses))
        m["n_be"] = int(len(bes)); m["win_rate"] = float(len(wins)/len(trades))
        m["avg_win"] = float(wins.pnl_acct.mean()) if len(wins) else 0.0
        m["avg_loss"] = float(losses.pnl_acct.mean()) if len(losses) else 0.0
        gross_win = float(wins.pnl_acct.sum()) if len(wins) else 0.0
        gross_loss = float(-losses.pnl_acct.sum()) if len(losses) else 1e-9
        m["profit_factor"] = gross_win / max(gross_loss, 1e-9)
        m["expectancy_acct"] = float(pnl.mean()); m["total_pnl"] = float(pnl.sum())
        r = trades["r_multiple"].astype(float)
        m["mean_R"] = float(r.mean()); m["median_R"] = float(r.median())
        m["std_R"] = float(r.std()); m["max_win_R"] = float(r.max()); m["max_loss_R"] = float(r.min())
        if not eq.empty and "equity" in eq.columns:
            eq_s = eq["equity"].astype(float)
            m["final_equity"] = float(eq_s.iloc[-1])
            m["return_pct"] = float((eq_s.iloc[-1]-self.bt.starting_equity)/self.bt.starting_equity*100)
            running_max = eq_s.cummax(); dd = eq_s - running_max
            m["max_drawdown_acct"] = float(dd.min())
            m["max_drawdown_pct"] = float((dd/running_max).min()*100) if running_max.iloc[-1] > 0 else 0.0

        def _breakdown(key):
            m[key] = {}
            if key not in trades.columns:
                return
            for v in trades[key].dropna().unique():
                sub = trades[trades[key] == v]
                if len(sub) == 0: continue
                sw = sub[sub.pnl_acct > 0]; sl_ = sub[sub.pnl_acct < 0]
                entry = {"n": int(len(sub))}
                if len(sw): entry["win_rate"] = float(len(sw)/len(sub))
                entry["mean_R"] = float(sub.r_multiple.mean())
                entry["PF"] = float(sw.pnl_acct.sum()/max(-sl_.pnl_acct.sum(),1e-9)) if len(sl_) else float("inf")
                if "pnl_acct" in sub.columns:
                    entry["total_pnl"] = float(sub.pnl_acct.sum())
                m[key][str(v)] = entry

        for key in ["by_grade", "by_session", "by_zone", "by_direction", "by_killzone", "by_ob_tf", "by_exit"]:
            k = {"by_grade":"grade","by_session":"session","by_zone":"zone_kind",
                 "by_direction":"direction","by_killzone":"killzone","by_ob_tf":"ob_tf",
                 "by_exit":"close_reason"}[key]
            _breakdown(k); m[key] = m.pop(k)
        if "close_reason" in trades.columns:
            m["exit_reasons"] = trades.close_reason.value_counts().to_dict()
        return m


def run_backtest(
    data_cfg: DataConfig, raw_symbol: str, signals_df: pd.DataFrame,
    account: AccountSpec | None = None, bt_cfg: BacktestConfig | None = None,
    strat_cfg: StrategyConfig | None = None, symbol_overrides: dict | None = None,
    progress: ProgressFn | None = None,
) -> BacktestResult:
    spec = spec_for_symbol(raw_symbol, symbol_overrides)
    acct = account or AccountSpec()
    if spec.currency_profit not in acct.fx_to_account:
        acct.fx_to_account[spec.currency_profit] = 1.0 if spec.currency_profit == acct.currency else 18.5
    engine = BacktestEngine(spec, acct, strat_cfg, bt_cfg, progress)
    base = data_cfg.aligned_path / f"symbol={raw_symbol}" / "timeframe=M1"
    files = sorted(Path(base).glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No aligned M1 parquets at {base}")
    return engine.run(files, signals_df)
