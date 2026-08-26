"""Layer 6-ready LIVE trading loop for SlyTrade v0.9.2 scalper persona.

Connects to MT5 via the mt5linux RPyC bridge (run `bash start_mt5_bridge.sh`
in another terminal first), pulls multi-timeframe bars, computes Layer 2
features, performs causal MTF alignment, runs the Layer 4/5 signal scanner
statefully, and when a signal fires sends a market order with SL/TP sized
per our grade-tiered risk rules.

Open positions are monitored every 5 seconds for:
  - SL hit (broker-side, redundant check via equity diff)
  - TP hit (broker-side)
  - M15 CHoCH emergency exit (closes at market if detected)
  - Time-stop (4 hours / 240 M1 bars flat regardless of P&L)

Run:
    bash start_mt5_bridge.sh        # in another terminal
    python -m slytrade.live.trader --symbol XAUUSDm            # dry-run champion
    python -m slytrade.live.trader --symbol XAUUSDm --all --verbose  # see ALL signals
    python -m slytrade.live.trader --symbol XAUUSDm --live     # real trading (champion)

Default persona: v0.9.2 champion (longs-only, 0.85R one-shot, >=2 ATR stops,
grades A+/A/B, M5+M15 OBs, London/NY). Use --all to switch to the unrestricted
scalper persona (long+short, all grades, H1+M15+M5 OBs+FVGs, LIQ_SWEEP + BOS_CONT
quick scalps, Asian+off-hours unlocked, persona_gating=False) so you see
EVERYTHING the engine fires -- including the money-printing liquidity grabs
and momentum bursts the champion filters out.
"""
from __future__ import annotations

import argparse
import signal
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Imports from our own Layer 2-5 stack
# --------------------------------------------------------------------------- #
from ..backtest.specs import AccountSpec, SymbolSpec, spec_for_symbol
from ..data.features import DEFAULT_CONFIG, process_bars
from ..data.mtf_align import _asof_merge, _prep_htf_frame
from ..data.schemas import normalize_bar_frame
from ..data.time import timeframe_timedelta
from ..strategy.config import StrategyConfig, champion_persona, rl_training_persona
from ..strategy.signals import _evaluate_row

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TIMEFRAME_ATTRS = {
    "M1":  "TIMEFRAME_M1",
    "M5":  "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1":  "TIMEFRAME_H1",
    "H4":  "TIMEFRAME_H4",
    "D1":  "TIMEFRAME_D1",
}
MAGIC = 260810  # SlyTrade magic number
# Per-TF history windows — GENEROUS by design so the engine sees EVERYTHING.
#
# Rule per TF: 3 × EMA200 lookback for full indicator warmup, plus enough
# swing history for ATR-ZigZag major pivots (mult=4 ATR) to form cleanly,
# plus enough historical swing reference levels that liquidity sweeps
# against weeks-old levels are still detectable.
#
# M1 is intentionally the LARGEST window — that's where quick scalps fire,
# and we do NOT want a swing low from last week to fall out of the buffer
# right as price comes back to sweep it. Total processing per cycle
# benchmarks at ~1s on a modern CPU — well within our 60-second poll budget.
#
#                   bars         ~calendar days   why
WARMUP_BARS = {
    "M1":  60000,  # ~42d  (6 wks)   swing refs + liq levels weeks back
    "M5":  15000,  # ~52d  (7.5 wks) trigger TF: full multi-week structure
    "M15":  6000,  # ~62d  (9 wks)   OB TF + A+ premium/discount zone
    "M30":  3000,  # ~62d  (9 wks)   mid-TF confluence for grade/runner
    "H1":   2400,  # ~100d (14 wks)  OB TF + runners; H1 EMA200 needs ~12d
    "H4":   1200,  # ~200d (28 wks)  HTF bias for A+; H4 EMA200 needs ~50 days
    "D1":    500,  # ~500d (~1.4y)   D1 EMA200 = 200d; yearly hi/lo runners
}
HTFS = ["M5", "M15", "M30", "H1", "H4", "D1"]
CHOPCH_EMERGENCY_TF = "M15"
TIME_STOP_BARS = 240


# --------------------------------------------------------------------------- #
# Live MT5 helpers
# --------------------------------------------------------------------------- #

def connect_mt5(host: str = "127.0.0.1", port: int = 18812) -> Any:
    """Connect to MT5 via mt5linux's RPyC bridge (Wine side)."""
    from mt5linux import MetaTrader5  # type: ignore
    mt5 = MetaTrader5(host=host, port=port)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    return mt5


def _to_dict(obj: Any) -> dict:
    """Bridge-proof attribute dict (RPyC can return SimpleNamespace or dict)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        return obj._asdict()
    except Exception:
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


def resolve_symbol_spec(mt5: Any, symbol: str, account_ccy: str, usd_zar: float) -> tuple[str, SymbolSpec]:
    """Resolve the broker's actual symbol name (handles 'm' suffix etc.) and build SymbolSpec."""
    all_syms = mt5.symbols_get() or []
    names: list[str] = []
    for s in all_syms:
        d = _to_dict(s)
        n = d.get("name", "")
        if n:
            names.append(str(n))
    target = symbol.lower()
    def rank(n: str) -> tuple:
        low = n.lower()
        return (low != target, "247" in low, len(n), n)
    candidates = sorted((n for n in names if target in n.lower()), key=rank)
    if not candidates:
        raise RuntimeError(f"No MT5 symbol matching '{symbol}'. Available: {sorted(names)[:30]}")
    resolved = candidates[0]
    mt5.symbol_select(resolved, True)
    info = _to_dict(mt5.symbol_info(resolved))
    overrides = {
        "point": float(info.get("point", 0.001)),
        "digits": int(info.get("digits", 3)),
        "contract_size": float(info.get("trade_contract_size", 100.0)),
        "volume_min": float(info.get("volume_min", 0.01)),
        "volume_max": float(info.get("volume_max", 100.0)),
        "volume_step": float(info.get("volume_step", 0.01)),
        "currency_profit": str(info.get("currency_profit", "USD")),
    }
    spec = spec_for_symbol(resolved, overrides=overrides)
    return resolved, spec


def fetch_bars(mt5: Any, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Fetch `count` COMPLETED bars ending 'now' from MT5, normalized.

    MT5 copy_rates_from_pos(symbol, tf, 0, N) returns N bars where the most
    recent (last after time-sort) is the CURRENTLY-FORMING bar — its OHLC is
    still mutating. We over-fetch and drop that in-progress bar so our feature
    pipeline only sees closed bars (causality-safe).
    """
    tf_const = getattr(mt5, TIMEFRAME_ATTRS[timeframe])
    want = int(count) + 2  # +1 for in-progress bar, +1 safety margin
    raw = mt5.copy_rates_from_pos(symbol, tf_const, 0, want)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = normalize_bar_frame(raw, symbol, timeframe)
    if df.empty:
        return df
    # Drop the in-progress bar: any bar whose close time (time + tf_duration)
    # hasn't passed yet is incomplete.
    dur = timeframe_timedelta(timeframe)
    now = datetime.now(UTC)
    df = df[df["time"] + dur <= now].copy()
    # Keep only the last `count` completed bars
    if len(df) > count:
        df = df.iloc[-count:].copy()
    return df.reset_index(drop=True)


def fetch_and_process_tf(mt5: Any, symbol: str, tf: str, count: int) -> pd.DataFrame:
    """Fetch raw bars from MT5 and compute Layer-2 features for one TF."""
    raw = fetch_bars(mt5, symbol, tf, count)
    if raw.empty:
        return raw
    return process_bars(raw, tf, DEFAULT_CONFIG)


# --------------------------------------------------------------------------- #
# Causal MTF alignment (mini version of data/mtf_align.py for live frames)
# --------------------------------------------------------------------------- #

def align_live(m1: pd.DataFrame, htf_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Causally align HTF feature frames onto M1 bars (strict backward asof)."""
    df = m1.copy().sort_values("time").reset_index(drop=True)
    for tf, htf in htf_frames.items():
        if htf.empty:
            continue
        dur = timeframe_timedelta(tf)
        htf = htf.copy()
        htf["decision_time"] = htf["time"] + dur
        prepped = _prep_htf_frame(htf, tf)
        df = _asof_merge(df, prepped, tf)
    return df


# --------------------------------------------------------------------------- #
# Live position tracking
# --------------------------------------------------------------------------- #

@dataclass
class LiveTrade:
    ticket: int
    direction: int           # +1 long, -1 short
    entry: float
    sl: float
    tp: float
    lots: float
    open_time: datetime
    grade: str
    risk_pct: float
    bars_held: int = 0
    pnl: float = 0.0
    close_price: float | None = None
    close_reason: str | None = None
    closed: bool = False


class LiveTrader:
    def __init__(
        self,
        mt5: Any,
        symbol: str,
        spec: SymbolSpec,
        cfg: StrategyConfig,
        acct: AccountSpec,
        *,
        live: bool = False,
        risk_cap: float = 0.02,
        max_open: int = 3,
        poll_interval: float = 5.0,
        verbose: bool = False,
    ):
        self.mt5 = mt5
        self.symbol = symbol
        self.spec = spec
        self.cfg = cfg
        self.acct = acct
        self.live = live
        self.risk_cap = risk_cap
        self.max_open = max_open
        self.poll_interval = poll_interval
        self.verbose = verbose

        self._state: dict = {}
        self._trades: dict[int, LiveTrade] = {}
        self._last_processed_m1_time: datetime | None = None
        self._signals_fired: set[str] = set()   # dedupe keys
        self._cycle = 0
        self._warmed_up = False

    # ------------------------------------------------------------------ #
    # Account info
    # ------------------------------------------------------------------ #
    def account(self) -> dict:
        return _to_dict(self.mt5.account_info())

    def equity(self) -> float:
        return float(self.account().get("equity", 0.0))

    def quote(self) -> tuple[float, float]:
        """Return (bid, ask). If MT5 returns stale/zero ticks (can happen
        briefly right after a large historical pull or on session close),
        return the last good quote so the status line and order sizing
        don't see 0.0 which wrecks margin/risk math."""
        t = _to_dict(self.mt5.symbol_info_tick(self.symbol))
        bid = float(t.get("bid", 0.0))
        ask = float(t.get("ask", 0.0))
        if bid > 0 and ask > 0 and ask >= bid:
            self._last_quote = (bid, ask)
            return bid, ask
        return getattr(self, "_last_quote", (bid, ask))

    # ------------------------------------------------------------------ #
    # Order entry
    # ------------------------------------------------------------------ #
    def _place_market(self, direction: int, lots: float, sl: float, tp: float,
                      comment: str) -> int | None:
        """Place a market order. Returns ticket or None."""
        bid, ask = self.quote()
        price = ask if direction == 1 else bid
        digits = self.spec.digits
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": round(float(lots), 2),
            "type": self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": round(float(sl), digits),
            "tp": round(float(tp), digits),
            "deviation": 30,
            "magic": MAGIC,
            "comment": comment[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        if not self.live:
            print(f"    [DRY-RUN] {'BUY' if direction==1 else 'SELL'} {lots} {self.symbol} @ {price:.{digits}f}  SL={sl:.{digits}f}  TP={tp:.{digits}f}  ({comment})")
            return -int(time.time() * 1000)   # fake ticket
        res = _to_dict(self.mt5.order_send(req))
        retcode = int(res.get("retcode", -1))
        DONE = {int(getattr(self.mt5, "TRADE_RETCODE_DONE", 10009)),
                int(getattr(self.mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))}
        if retcode in DONE:
            return int(res.get("order", 0)) or int(res.get("deal", 0))
        print(f"    [REJECT] {retcode} {res.get('comment','')} req={req}")
        return None

    def _close_position(self, ticket: int, reason: str) -> bool:
        """Close an open position by ticket at market."""
        pos = self._get_position(ticket)
        if pos is None:
            return False
        vol = float(pos.get("volume", 0.0))
        ptype = int(pos.get("type", 0))
        close_type = self.mt5.ORDER_TYPE_SELL if ptype == 0 else self.mt5.ORDER_TYPE_BUY
        bid, ask = self.quote()
        price = bid if ptype == 0 else ask
        digits = self.spec.digits
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": round(vol, 2),
            "type": close_type,
            "position": int(ticket),
            "price": price,
            "deviation": 30,
            "magic": MAGIC,
            "comment": reason[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        if not self.live:
            lt = self._trades.get(int(ticket))
            if lt:
                lt.closed = True; lt.close_reason = reason; lt.close_price = price
            print(f"    [DRY-RUN] CLOSE ticket={ticket} @ {price:.{digits}f} ({reason})")
            return True
        res = _to_dict(self.mt5.order_send(req))
        retcode = int(res.get("retcode", -1))
        DONE = {int(getattr(self.mt5, "TRADE_RETCODE_DONE", 10009)),
                int(getattr(self.mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))}
        return retcode in DONE

    def _get_position(self, ticket: int) -> dict | None:
        positions = self.mt5.positions_get(ticket=int(ticket)) or []
        for p in positions:
            d = _to_dict(p)
            if int(d.get("ticket", -1)) == int(ticket):
                return d
        return None

    def _our_open_positions(self) -> dict[int, dict]:
        """Return MT5 positions opened by us (magic number match)."""
        out: dict[int, dict] = {}
        for p in (self.mt5.positions_get(symbol=self.symbol) or []):
            d = _to_dict(p)
            if int(d.get("magic", 0)) == MAGIC:
                out[int(d["ticket"])] = d
        return out

    # ------------------------------------------------------------------ #
    # Core signal handling
    # ------------------------------------------------------------------ #
    def _vlog(self, msg: str) -> None:
        if self.verbose:
            print(f"    [SIG] {msg}")

    def _handle_signal(self, sig) -> None:
        zone_id = sig.ob_tf or (f"fvg{sig.fvg_top:.0f}" if sig.fvg_top else "-")
        key = f"{sig.time.isoformat()}|{sig.direction}|{sig.grade}|{sig.setup_kind}|{zone_id}"
        side = "LONG" if sig.direction == 1 else "SHORT"
        setup = getattr(sig, 'setup_kind', 'RETEST_OB')
        if key in self._signals_fired:
            self._vlog(f"dupe {side} {setup}/{sig.grade} @ {sig.time} — skip")
            return
        # Directional safety net (engine already filters when persona_gating=True)
        if sig.direction == -1 and not self.cfg.confluence.accept_shorts:
            self._vlog(f"{side} {setup} blocked by accept_shorts=False"); self._signals_fired.add(key); return
        if sig.direction == 1 and not self.cfg.confluence.accept_longs:
            self._vlog(f"{side} {setup} blocked by accept_longs=False"); self._signals_fired.add(key); return
        open_n = len(self._our_open_positions()) + sum(1 for t in self._trades.values() if not t.closed)
        if open_n >= self.max_open:
            self._vlog(f"{side} {setup}/{sig.grade} rejected: max_open={self.max_open} reached (open={open_n})")
            return
        equity = self.equity()
        bid, ask = self.quote()
        entry_approx = ask if sig.direction == 1 else bid
        half_spread = (ask - bid) * 0.5
        slip_pts = 5 * self.spec.point
        fill = entry_approx + (half_spread + slip_pts) if sig.direction == 1 else entry_approx - (half_spread + slip_pts)
        risk_per_unit = abs(fill - float(sig.stop))
        if risk_per_unit <= 0:
            self._vlog(f"{side} {setup}/{sig.grade} rejected: invalid risk_per_unit={risk_per_unit}")
            return
        risk_pct = min(float(sig.risk_pct), self.risk_cap)
        risk_acct = risk_pct * equity
        risk_quote = risk_acct / self.acct.fx_to_account.get(self.spec.currency_profit, 1.0)
        lots = self.spec.lots_for_risk(risk_per_unit, risk_quote)
        if lots < self.spec.volume_min - 1e-9:
            self._vlog(f"{side} {setup}/{sig.grade} rejected: lots={lots:.4f} < vol_min={self.spec.volume_min} (eq={equity:.0f} risk_pct={risk_pct:.4f})")
            return
        tp = fill + sig.direction * self.cfg.exits.tp1_r * risk_per_unit
        sl = float(sig.stop)
        margin_quote = (lots * self.spec.contract_size * fill) / max(self.acct.leverage, 1)
        margin_acct = self.acct.to_account_ccy(margin_quote, self.spec.currency_profit)
        if margin_acct > equity * 0.95:
            self._vlog(f"{side} {setup}/{sig.grade} rejected: margin {margin_acct:.0f} > 95% equity {equity:.0f}")
            return
        zone_label = sig.ob_tf or (f"FVG@{sig.fvg_top:.0f}" if sig.fvg_top else "")
        self._vlog(f"{side} {setup}/{sig.grade} {zone_label} kz={sig.killzone} fill={fill:.{self.spec.digits}f} "
                   f"sl={sl:.{self.spec.digits}f} tp={tp:.{self.spec.digits}f} lots={lots:.2f} risk={risk_pct*100:.2f}%")
        comment = f"L5 {sig.grade} {setup} {sig.killzone}"
        ticket = self._place_market(sig.direction, lots, sl, tp, comment)
        if ticket is None:
            self._vlog(f"{side} {setup}/{sig.grade} ORDER REJECTED by broker")
            return
        self._signals_fired.add(key)
        self._trades[ticket] = LiveTrade(
            ticket=ticket, direction=sig.direction, entry=fill, sl=sl, tp=tp,
            lots=lots, open_time=datetime.now(UTC), grade=sig.grade, risk_pct=risk_pct,
        )
        print(f"    [ENTRY] ticket={ticket} {side} {setup} {lots} lots @ {fill:.{self.spec.digits}f} "
              f"grade={sig.grade} {zone_label} kz={sig.killzone} SL={sl:.{self.spec.digits}f} TP={tp:.{self.spec.digits}f}")

    # ------------------------------------------------------------------ #
    # Position monitoring
    # ------------------------------------------------------------------ #
    def _monitor_positions(self, latest_m1_row: pd.Series | None, *, new_bar: bool = False) -> None:
        """Check SL/TP/emergency/time-stop for open positions.

        new_bar=True means this call corresponds to a fresh closed M1 bar —
        that's where bars_held increments and CHoCH/time-stop checks run.
        Intra-bar poll calls (new_bar=False) only check broker-side SL/TP.
        """
        bid, ask = self.quote()

        # ---------- SL / TP ---------- #
        if not self.live:
            to_close: list[tuple[int, str]] = []
            for ticket, lt in list(self._trades.items()):
                if lt.closed:
                    continue
                price = bid if lt.direction == 1 else ask
                hit_sl = (lt.direction == 1 and price <= lt.sl) or (lt.direction == -1 and price >= lt.sl)
                hit_tp = (lt.direction == 1 and price >= lt.tp) or (lt.direction == -1 and price <= lt.tp)
                if hit_sl:   to_close.append((ticket, "SL"))
                elif hit_tp: to_close.append((ticket, "TP1"))
                # rough running P&L for status line
                r_mult = (price - lt.entry) / max(abs(lt.tp - lt.entry), 1e-9)
                if lt.direction == -1:
                    r_mult = -r_mult
                lt.pnl = r_mult * lt.risk_pct * self.equity()
            for ticket, reason in to_close:
                lt = self._trades[ticket]
                price = bid if lt.direction == 1 else ask
                lt.close_price = price; lt.closed = True; lt.close_reason = reason
                if self._close_position(ticket, reason):
                    print(f"    [EXIT] ticket={ticket} reason={reason}")
            broker_pos: dict[int, dict] = {}
        else:
            broker_pos = self._our_open_positions()
            broker_tickets = set(broker_pos.keys())
            for ticket, lt in list(self._trades.items()):
                if ticket not in broker_tickets and not lt.closed:
                    lt.closed = True
                    lt.close_reason = "BROKER"
                    print(f"    [EXIT] ticket={ticket} reason=BROKER (SL/TP hit)")

        # ---------- Per-bar checks (CHoCH emergency + time stop) ---------- #
        if not new_bar:
            return
        for ticket, lt in list(self._trades.items()):
            if lt.closed:
                continue
            if self.live and ticket not in broker_pos:
                continue
            lt.bars_held += 1

            # M15 CHoCH emergency exit
            if latest_m1_row is not None:
                if lt.direction == 1 and bool(latest_m1_row.get(f"{CHOPCH_EMERGENCY_TF}_major_choch_dn", False)):
                    print(f"    [EMERGENCY M15 CHoCH DOWN against LONG ticket={ticket}] — closing")
                    self._close_position(ticket, "M15_CHOCH"); continue
                if lt.direction == -1 and bool(latest_m1_row.get(f"{CHOPCH_EMERGENCY_TF}_major_choch_up", False)):
                    print(f"    [EMERGENCY M15 CHoCH UP against SHORT ticket={ticket}] — closing")
                    self._close_position(ticket, "M15_CHOCH"); continue

            # Time-stop: backtest uses time_stop_min_r=0.0 => FLAT after
            # TIME_STOP_BARS regardless of P&L (parity with backtest engine).
            if lt.bars_held >= TIME_STOP_BARS:
                if self.live:
                    pos = broker_pos.get(ticket, {})
                    cur_profit = float(pos.get("profit", 0.0))
                    price = bid if lt.direction == 1 else ask
                    r_dist = (price - lt.entry) if lt.direction == 1 else (lt.entry - price)
                    r_mult = r_dist / max(abs(lt.tp - lt.entry), 1e-9)
                    print(f"    [TIME-STOP] ticket={ticket} bars={lt.bars_held} profit={cur_profit:.2f} r={r_mult:+.2f} — closing")
                else:
                    print(f"    [TIME-STOP] ticket={ticket} bars={lt.bars_held} — closing (dry-run)")
                self._close_position(ticket, "TIME_STOP")

    # ------------------------------------------------------------------ #
    # One M1 cycle
    # ------------------------------------------------------------------ #
    def _prime_state(self, aligned: pd.DataFrame) -> None:
        """Walk ALL completed historical bars to build zone/trigger state.

        After this call the state machine contains the correct set of active
        OBs/FVGs, swing pivots, CHoCH/BOS markers, trigger timestamps as of the
        most recent completed bar. No trades are opened during warmup.
        """
        self._state.clear()
        n = len(aligned)
        n_err = 0
        for i in range(n):
            row = aligned.iloc[i]
            try:
                _evaluate_row(int(i), row, self.cfg, self._state)
            except Exception:
                n_err += 1
        if n > 0:
            self._last_processed_m1_time = aligned.iloc[-1]["time"]
        self._warmed_up = True
        if self.verbose:
            print(f"  [warmup] processed {n} bars ({n_err} suppressed errors)")
            self._dump_state("post-warmup")

    def _dump_state(self, tag: str) -> None:
        """Print active zones and trigger state for debugging."""
        active_zones: list[str] = []
        trigger_ts: list[str] = []
        sweep_px_lines: list[str] = []
        now_ts = pd.Timestamp.now(tz=UTC)
        for k, v in self._state.items():
            if k.startswith("_last_"):
                if isinstance(v, pd.Timestamp) or (v is not None and not isinstance(v, (int, float))):
                    # timestamp entries (triggers, sweep events)
                    try:
                        age_s = (now_ts - pd.Timestamp(v, tz=UTC)).total_seconds()
                        age_min = age_s / 60.0
                        trigger_ts.append(f"{k}={pd.Timestamp(v).strftime('%H:%M:%S')} (age={age_min:.0f}m)")
                    except Exception:
                        trigger_ts.append(f"{k}={v}")
                    continue
                if isinstance(v, (int, float)) and k.endswith("_px"):
                    # sweep price levels (floats, not timestamps)
                    try:
                        sweep_px_lines.append(f"  {k}={float(v):.2f}")
                    except Exception:
                        sweep_px_lines.append(f"  {k}={v}")
                    continue
            if isinstance(v, dict) and not k.endswith("_entered") and not v.get("mitigated", False):
                try:
                    top = v.get("top", float("nan")); bot = v.get("bot", float("nan"))
                    active_zones.append(f"{k} top={top:.1f} bot={bot:.1f}")
                except Exception:
                    active_zones.append(f"{k} {dict(v)}")
        if trigger_ts:
            print(f"  [{tag}] triggers:")
            for t in trigger_ts[:20]:
                print(f"    {t}")
        if sweep_px_lines:
            print(f"  [{tag}] sweep extremes:")
            for line in sweep_px_lines:
                print(line)
        print(f"  [{tag}] active zones ({len(active_zones)}):")
        for z in active_zones[:20]:
            print(f"    {z}")
        if len(active_zones) > 20:
            print(f"    ... and {len(active_zones)-20} more")

    def _structure_diagnostics(self, aligned: pd.DataFrame) -> None:
        """Print counts of recent displacements/BOS/CHoCH/liq-sweeps on trigger
        TFs so we can see whether features are firing during big moves.

        HTF boolean columns are forward-filled across M1 bars by the causal
        asof merge (one M5 bar maps to 5 M1 rows, etc.), so a raw ``sum()``
        overcounts. We count *distinct HTF-bar firings* by masking to rows
        where the HTF ``{tf}_bar_time`` just changed (the first M1 bar of
        a new HTF candle) — equivalent to counting HTF events, not M1 echoes.
        """
        # Pre-compute first-bar masks for each HTF (True on the first M1 row
        # of a new HTF candle, where bar_time differs from previous M1 row).
        first_bar: dict[str, np.ndarray] = {}
        for tf in ("M5", "M15", "M30", "H1", "H4"):
            bt_col = f"{tf}_bar_time"
            if bt_col in aligned.columns:
                bt = pd.to_datetime(aligned[bt_col], utc=True, errors="coerce")
                changed = np.concatenate(([True], bt.diff().dt.total_seconds().iloc[1:].fillna(0).abs().to_numpy() > 0))
                first_bar[tf] = changed
            else:
                first_bar[tf] = np.ones(len(aligned), dtype=bool)

        def _events(tail_idx: np.ndarray, tf: str, col: str) -> tuple[int, pd.Timestamp | None]:
            full_col = col if tf == "M1" else f"{tf}_{col}"
            if full_col not in aligned.columns:
                return 0, None
            vals = aligned[full_col].iloc[tail_idx].fillna(False).astype(bool).to_numpy()
            if tf == "M1":
                mask = np.ones(len(vals), dtype=bool)
            else:
                mask = first_bar[tf][tail_idx]
            hits = vals & mask
            n = int(hits.sum())
            last_t: pd.Timestamp | None = None
            if n > 0:
                # last index where hits is True (walk tail backwards)
                for k in range(len(hits) - 1, -1, -1):
                    if hits[k]:
                        last_t = aligned["time"].iloc[tail_idx[k]]
                        break
            return n, last_t

        for tf, lookback in [("M1", 30), ("M5", 60), ("M15", 240), ("M30", 480), ("H1", 600)]:
            tail_start = max(0, len(aligned) - lookback)
            tail_idx = np.arange(tail_start, len(aligned))
            counts: dict[str, int] = {}
            last_ts: dict[str, pd.Timestamp | None] = {}
            for ev in ("bull_disp", "bear_disp",
                       "minor_bos_up", "minor_bos_dn", "minor_choch_up", "minor_choch_dn",
                       "major_bos_up", "major_bos_dn", "major_choch_up", "major_choch_dn",
                       "bull_liq_sweep", "bear_liq_sweep"):
                n, t = _events(tail_idx, tf, ev)
                if n > 0:
                    counts[ev] = n
                    last_ts[ev] = t
            def _fmt(ev: str, _lt: dict = last_ts) -> str:
                t = _lt.get(ev)
                return pd.Timestamp(t).strftime("%H:%M") if t is not None else "None"
            if counts:
                print(f"  [diag] {tf} last {lookback}M1: {counts} "
                      f"bull_disp={_fmt('bull_disp')} bear_disp={_fmt('bear_disp')} "
                      f"bull_sweep={_fmt('bull_liq_sweep')} bear_sweep={_fmt('bear_liq_sweep')}")

    def _cycle_fn(self) -> None:
        self._cycle += 1
        m1_raw = fetch_bars(self.mt5, self.symbol, "M1", WARMUP_BARS["M1"])
        if m1_raw.empty or len(m1_raw) < 300:
            print(f"[cycle {self._cycle}] not enough M1 data yet ({len(m1_raw)} bars)")
            return
        htf_processed: dict[str, pd.DataFrame] = {}
        for tf in HTFS:
            htf_processed[tf] = fetch_and_process_tf(self.mt5, self.symbol, tf, WARMUP_BARS[tf])
        m1 = process_bars(m1_raw, "M1", DEFAULT_CONFIG)
        aligned = align_live(m1, htf_processed)
        if aligned.empty:
            return
        n = len(aligned)

        # First cycle: walk ALL history to prime state so OBs that formed
        # before we started are tracked correctly. Without this, zones appear
        # "out of thin air" and never trigger entries (the bug that caused
        # zero signals during the sell-off).
        if not self._warmed_up:
            self._prime_state(aligned)
            print(f"  Warmed up on {n} M1 bars — state primed, watching for new signals ...")

        # Only process bars newer than the last one we already evaluated.
        new_mask = pd.Series(True, index=aligned.index)
        if self._last_processed_m1_time is not None:
            new_mask = aligned["time"] > self._last_processed_m1_time
        new_rows = aligned.index[new_mask].tolist()

        n_sigs_this_cycle = 0
        for i in new_rows:
            row = aligned.iloc[i]
            try:
                sig = _evaluate_row(int(i), row, self.cfg, self._state)
            except Exception:
                sig = None
            if sig is not None:
                self._handle_signal(sig)
                n_sigs_this_cycle += 1

        if new_rows:
            self._last_processed_m1_time = aligned.iloc[new_rows[-1]]["time"]

        if self.verbose and (n_sigs_this_cycle > 0 or self._cycle % 5 == 0):
            bid, ask = self.quote()
            self._dump_state(f"cycle {self._cycle} bid={bid:.1f} new_bars={len(new_rows)} sigs={n_sigs_this_cycle}")
            self._structure_diagnostics(aligned)
            lb = aligned.iloc[-1]
            flags: list[str] = []
            for col in ("M5_bull_disp", "M5_bear_disp",
                        "M5_minor_bos_up", "M5_minor_bos_dn",
                        "M5_minor_choch_up", "M5_minor_choch_dn",
                        "M5_major_bos_up", "M5_major_bos_dn",
                        "M5_major_choch_up", "M5_major_choch_dn",
                        "M15_major_choch_up", "M15_major_choch_dn"):
                if col in aligned.columns and bool(lb.get(col, False)):
                    flags.append(col.replace("M5_", "").replace("M15_", "M15:"))
            if flags:
                print(f"  [latest M1 bar flags] {', '.join(flags)}")

        latest = aligned.iloc[-1]
        self._monitor_positions(latest, new_bar=True)
        self._print_status(latest)

    def _print_status(self, latest: pd.Series) -> None:
        bid, ask = self.quote()
        eq = self.equity()
        acc = self.account()
        n_open = len(self._our_open_positions()) if self.live else sum(1 for t in self._trades.values() if not t.closed)
        closed = [t for t in self._trades.values() if t.closed]
        wins = sum(1 for t in closed if t.pnl > 0) if closed else 0
        total_pnl = sum(t.pnl for t in closed) if closed else 0.0
        floating = 0.0
        if self.live:
            for p in self._our_open_positions().values():
                floating += float(p.get("profit", 0.0)) * self.acct.fx_to_account.get(self.spec.currency_profit, 1.0)
        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        bar_time = pd.to_datetime(latest["time"]).strftime("%H:%M") if "time" in latest.index else "?"
        print(
            f"[{now_str}] M1 {bar_time}  bid={bid:.{self.spec.digits}f} ask={ask:.{self.spec.digits}f}  "
            f"eq={eq:.2f}{acc.get('currency','ZAR')}  open={n_open}  floating={floating:+.2f}  "
            f"closed={len(closed)} wins={wins} realized={total_pnl:+.2f}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        persona_label = "v0.9.2 SCALPER (all setups, RL-unrestricted)" if not self.cfg.confluence.persona_gating else "v0.9.2 champion (long-only A+/A/B RETEST_OB)"
        print(f"SlyTrade LIVE v0.9.2  symbol={self.symbol}  live={self.live}  risk_cap={self.risk_cap*100:.1f}%  max_open={self.max_open}")
        print(f"persona: {persona_label}")
        print("setups : RETEST_OB RETEST_FVG LIQ_SWEEP BOS_CONT  (champion gates apply unless --all)")
        print(f"Magic={MAGIC}")
        print("-"*80)
        running = {"v": True}
        def _stop(signum, frame):
            running["v"] = False
            print("\n[STOP] shutting down ...")
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        try:
            self._cycle_fn()
        except Exception as e:
            print(f"[ERROR] initial cycle: {e}")
            traceback.print_exc()
        while running["v"]:
            now = datetime.now(UTC)
            next_min = (now + timedelta(minutes=1)).replace(second=5, microsecond=0)
            sleep_s = (next_min - datetime.now(UTC)).total_seconds()
            if sleep_s < 0: sleep_s = 1.0
            end_wait = time.time() + sleep_s
            while running["v"] and time.time() < end_wait:
                if self._cycle % 6 == 0:
                    try:
                        self._monitor_positions(None)
                    except Exception:
                        pass
                time.sleep(min(self.poll_interval, end_wait - time.time()))
            if not running["v"]: break
            try:
                self._cycle_fn()
            except Exception as e:
                print(f"[ERROR] cycle: {e}")
                traceback.print_exc()
                time.sleep(10)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="SlyTrade v0.9.2 SCALPER LIVE trader (OB/FVG retests + liq sweeps + BOS continuation)")
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--live", action="store_true", help="Actually send orders (default: dry-run)")
    ap.add_argument("--risk-cap", type=float, default=0.01, help="Max risk per trade as fraction of equity (default 1%%)")
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--usd-zar", type=float, default=18.5)
    ap.add_argument("--leverage", type=int, default=2000)
    ap.add_argument("--verbose", action="store_true",
                    help="Dump zone/trigger state every 5 cycles and on every signal.")
    ap.add_argument("--all", dest="unrestricted", action="store_true",
                    help="Disable persona gating: emit ALL signals (long+short, all grades, "
                         "H1+M15+M5 OBs+FVGs, C-grades, Asian+off-hours) for RL-data collection "
                         "and pre-RL diagnostics. Default: champion persona (long-only, A+/A/B).")
    args = ap.parse_args()

    print("Connecting to MT5 bridge ...")
    mt5 = connect_mt5(args.host, args.port)
    acc = _to_dict(mt5.account_info())
    print(f"  login={acc.get('login')} server={acc.get('server')} balance={acc.get('balance')} "
          f"equity={acc.get('equity')} {acc.get('currency')} leverage={acc.get('leverage',args.leverage)}")

    resolved, spec = resolve_symbol_spec(mt5, args.symbol, str(acc.get("currency","ZAR")), args.usd_zar)
    print(f"  symbol resolved: {resolved} (point={spec.point} digits={spec.digits} "
          f"contract={spec.contract_size} vol_min={spec.volume_min} vol_step={spec.volume_step})")

    acct_spec = AccountSpec(
        starting_equity=float(acc.get("equity", 1000)),
        currency=str(acc.get("currency", "ZAR")),
        leverage=int(acc.get("leverage", args.leverage)),
        fx_to_account={"USD": args.usd_zar} if str(acc.get("currency","ZAR")) != "USD" else {"USD": 1.0},
    )
    cfg = rl_training_persona() if args.unrestricted else champion_persona()
    max_open_eff = args.max_open if not args.unrestricted else max(args.max_open, 10)
    print(f"  verbose       : {args.verbose}")
    print(f"  risk_cap      : {args.risk_cap*100:.1f}% per trade")
    print(f"  max_open      : {max_open_eff}")
    trader = LiveTrader(
        mt5=mt5, symbol=resolved, spec=spec, cfg=cfg, acct=acct_spec,
        live=args.live, risk_cap=args.risk_cap, max_open=max_open_eff,
        verbose=args.verbose,
    )
    try:
        trader.run()
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
