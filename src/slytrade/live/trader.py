"""Layer 6-ready LIVE trading loop for SlyTrade v0.9.15 hybrid-ladder persona.

Connects to MT5 via the mt5linux RPyC bridge (run `bash start_mt5_bridge.sh`
in another terminal first), pulls multi-timeframe bars, computes Layer 2
features, performs causal MTF alignment, runs the Layer 4/5 signal scanner
statefully, and when a signal fires sends a market or limit order with SL/TP
sized per our grade-tiered risk rules and dynamic working-lot sizing.

v0.9.15 changes:
  - Hybrid ladder exits: TP1 1.0R @ 50% → BE; TP2 2.5R @ 25%; runner 25% ATR trail + M5 CHoCH kill
  - working_lot default 0.04; risk_cap is only hard size rail (no 3× grade REJECT)
  - Limit-at-zone for RETEST/BREAKER with market fallback; market for DISP_TRAP/LIQ/BOS
  - One-shot re-arm when flat after TP
  - New setups: DISP_TRAP, BREAKER

Open positions are monitored every 5 seconds for:
  - SL hit (broker-side, redundant check via equity diff)
  - TP1/TP2 hit (partial close + BE/trail adjustment)
  - M5 CHoCH runner kill
  - M15 CHoCH emergency exit (closes at market if detected)
  - Time-stop (4 hours / 240 M1 bars flat regardless of P&L)

Run:
    bash start_mt5_bridge.sh        # in another terminal
    python -m slytrade.live.trader --symbol XAUUSDm            # dry-run champion
    python -m slytrade.live.trader --symbol XAUUSDm --all --verbose  # see ALL signals
    python -m slytrade.live.trader --symbol XAUUSDm --live     # real trading (champion)

Default persona: v0.9.15 champion (longs-only, hybrid ladder, >=2 ATR stops,
grades A+/A/B, M5+M15 OBs, London/NY). Use --all to switch to the unrestricted
scalper persona (long+short, all grades, H1+M15+M5 OBs+FVGs, DISP_TRAP + BREAKER +
LIQ_SWEEP + BOS_CONT, Asian+off-hours unlocked, persona_gating=False) so you see
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
    # Capture the broker's minimum stop distance (in points) so the live
    # trader can enforce it after SL clamping.  trade_stops_level is the
    # minimum distance from current price that SL/TP must be.
    stop_level_pts = int(info.get("trade_stops_level", 0))
    return resolved, spec, stop_level_pts


def fetch_bars(mt5: Any, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Fetch `count` COMPLETED bars ending 'now' from MT5, normalized.

    CRITICAL CAUSALITY RULE: MT5's copy_rates_from_pos(symbol, tf, 0, N) ALWAYS
    returns the CURRENTLY-FORMING bar as the LAST row -- its OHLC is mutating
    in real time and must NEVER be fed into the feature pipeline or signal
    engine. v0.9.7 guard:

      1. Wall-clock filter: keep bars whose close time has already passed
         on host clock (`time+dur <= now`). Same parity as v0.9.5 wall
         filter -- this tolerates arbitrary Wine/RPyC/NTP clock drift
         between host and MT5 server without filtering out legitimately
         closed bars (the first v0.9.7 build used `<= now - 2s` which
         filtered every bar on Sly's Pop-OS host when host clock lagged
         MT5 server time by ~3s under Wine, producing 0 bars / 'not
         enough M1 data yet' on every cycle). The forming bar's close is
         always >= now (it closes dur in the future) so it is excluded.
      2. For M1 ONLY: unconditionally drop the last row as belt-and-
         suspenders. M1 forming-bar OHLC poisons feature state
         (displacements, swings) within seconds, and a ~60s lag is
         acceptable for scalps. For M5+ the forming bar closes 5-1440
         minutes in the future, so the wall filter ALONE excludes it
         safely; an extra unconditional tail drop there (v0.9.5 bug)
         cut off the most recently CLOSED HTF bar, leaving structural
         flags (bull_disp, minor_choch_up, etc.) always one HTF period
         stale. v0.9.7 fixes that so HTF structure updates the moment
         the HTF bar closes.
    """
    tf_const = getattr(mt5, TIMEFRAME_ATTRS[timeframe])
    dur = timeframe_timedelta(timeframe)
    want = int(count) + 5
    raw = mt5.copy_rates_from_pos(symbol, tf_const, 0, want)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = normalize_bar_frame(raw, symbol, timeframe)
    if df.empty:
        return df
    # Wall-clock filter: drop bars whose close time has not yet passed on
    # host clock. This is the primary guard against the forming bar, whose
    # close time is always `dur` in the future (so it fails `<= now`).
    # Using `<= now` (v0.9.5 parity) not `< now - grace`: the latter caused
    # 0 bars on Sly's Pop-OS host when host clock was a few seconds behind
    # MT5 server time -- every bar's close looked "in the future" by more
    # than the grace window.
    now = datetime.now(UTC)
    df = df[df["time"] + dur <= now].copy()
    # M1 only: unconditional 1-bar tail drop as belt-and-suspenders.
    # Rationale: M1 forming-bar OHLC poisons displacements/swings within
    # seconds, and host-clock can be up to several seconds ahead of MT5
    # server under Wine, which lets the forming M1 bar slip past the wall
    # filter above. Dropping one more M1 bar costs ~60s latency which is
    # acceptable for scalps. For M5+ the forming bar's close is 5-1440
    # minutes in the future, so the wall filter ALONE excludes it safely
    # even with many seconds of clock skew; an extra tail drop there
    # (v0.9.5 bug) cut off the most recently CLOSED HTF bar, leaving
    # structural flags one HTF period stale. v0.9.7 fixes that.
    if timeframe == "M1" and len(df) > 1:
        df = df.iloc[:-1].copy()
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
    tp: float                # TP1 price
    lots: float
    open_time: datetime
    grade: str
    risk_pct: float
    bars_held: int = 0
    pnl: float = 0.0
    close_price: float | None = None
    close_reason: str | None = None
    closed: bool = False
    # v0.9.15 hybrid ladder state
    tp1_hit: bool = False    # TP1 reached, partial closed, SL moved to BE
    tp2_hit: bool = False    # TP2 reached, second partial closed
    tp2_price: float = 0.0   # TP2 target price
    runner_trail_px: float = 0.0  # current trailing stop for runner
    remaining_lots: float = 0.0  # lots remaining after partial closes
    original_lots: float = 0.0   # original lots at entry


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
        working_lot: float = 0.04,
        max_open: int = 3,
        poll_interval: float = 5.0,
        verbose: bool = False,
        stop_level_pts: int = 0,
    ):
        self.mt5 = mt5
        self.symbol = symbol
        self.spec = spec
        self.cfg = cfg
        self.acct = acct
        self.live = live
        self.risk_cap = risk_cap
        self.working_lot = working_lot
        self.max_open = max_open
        self.poll_interval = poll_interval
        self.verbose = verbose
        self._stop_level_pts = stop_level_pts

        self._state: dict = {}
        self._trades: dict[int, LiveTrade] = {}
        self._last_processed_m1_time: datetime | None = None
        self._signals_fired: set[str] = set()   # dedupe keys
        self._cycle = 0
        self._warmed_up = False
        self._starting_balance: float | None = None  # set after warmup
        self._last_broker_pos: dict[int, dict] = {}  # snapshot for P&L diffing
        self._wins_count: int = 0

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
    def _get_filling_modes(self) -> list[int]:
        """Determine supported filling modes for our symbol from broker info.

        symbol_info().filling_mode is a bitmask:
          bit 0 (1) = ORDER_FILLING_FOK
          bit 1 (2) = ORDER_FILLING_IOC
          bit 2 (4) = ORDER_FILLING_RETURN
        We try the most common ECN mode (IOC) first, then FOK, then RETURN.
        """
        info = _to_dict(self.mt5.symbol_info(self.symbol))
        mode_mask = int(info.get("filling_mode", 0))
        modes: list[int] = []
        # Try IOC first (Exness ECN), then FOK (standard), then RETURN (market-maker)
        candidates = [
            ("ORDER_FILLING_IOC", 2),
            ("ORDER_FILLING_FOK", 1),
            ("ORDER_FILLING_RETURN", 4),
        ]
        for attr, bit in candidates:
            val = int(getattr(self.mt5, attr, None) or 0)
            if mode_mask & bit:
                modes.append(val)
        # Fallback: if bitmask is 0 or we found nothing, try all three
        if not modes:
            modes = [int(getattr(self.mt5, a, 0)) for a in
                     ("ORDER_FILLING_IOC", "ORDER_FILLING_FOK", "ORDER_FILLING_RETURN")]
        return modes

    def _place_market(self, direction: int, lots: float, sl: float, tp: float,
                      comment: str) -> int | None:
        """Place a market order. Returns ticket or None."""
        bid, ask = self.quote()
        price = ask if direction == 1 else bid
        digits = self.spec.digits
        # Deviation: v0.9.7 used 30 points (= $0.03 on XAUUSDm) which is too
        # tight for London/NY volatility — a 50-cent spike during a sweep/BOS
        # is normal and produces retcode=-1 / "old price" rejects. Bump to
        # 500 points ($0.50 on XAU, ~5 pips on FX) which is tolerable slippage
        # on our 0.01-lot scalps (max 0.50 * 100 * 0.01 = $0.50 ≈ R9 extra).
        deviation = 500
        # v0.9.15.1: query broker's filling_mode bitmask and try supported modes
        # in order. Retries on retcode=-1 (RPyC bridge None) instead of giving
        # up on the first mode.
        filling_modes = self._get_filling_modes()
        DONE = {int(getattr(self.mt5, "TRADE_RETCODE_DONE", 10009)),
                int(getattr(self.mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))}
        RETRY_RETCODES = {10030, -1}  # unsupported fill / bridge None → try next mode
        if not self.live:
            print(f"    [DRY-RUN] {'BUY' if direction==1 else 'SELL'} {lots} {self.symbol} @ {price:.{digits}f}  SL={sl:.{digits}f}  TP={tp:.{digits}f}  ({comment})")
            return -int(time.time() * 1000)   # fake ticket
        for fill_mode in filling_modes:
            # Refresh price each attempt (stale price causes retcode=-1)
            bid, ask = self.quote()
            price = ask if direction == 1 else bid
            req = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": round(float(lots), 2),
                "type": self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": round(float(sl), digits),
                "tp": round(float(tp), digits),
                "deviation": deviation,
                "magic": MAGIC,
                "comment": comment[:31],
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode,
            }
            res = _to_dict(self.mt5.order_send(req))
            retcode = int(res.get("retcode", -1))
            if retcode in DONE:
                if fill_mode != filling_modes[0]:
                    fill_name = {1: "IOC", 2: "RETURN", 0: "FOK"}.get(fill_mode, str(fill_mode))
                    print(f"    [INFO] order filled with {fill_name} (previous modes failed)")
                return int(res.get("order", 0)) or int(res.get("deal", 0))
            if retcode in RETRY_RETCODES:
                # Brief pause before retry — bridge hiccups need a moment
                if retcode == -1:
                    time.sleep(0.2)
                continue
            # Definitive broker rejection (10016 invalid stops, etc.) — don't retry
            print(f"    [REJECT] {retcode} {res.get('comment','')} fill={fill_mode} req={req}")
            return None
        print(f"    [REJECT] all filling modes failed for {comment} (tried {len(filling_modes)} modes)")
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
        deviation = 500  # match _place_market
        DONE = {int(getattr(self.mt5, "TRADE_RETCODE_DONE", 10009)),
                int(getattr(self.mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010))}
        RETRY_RETCODES = {10030, -1}
        if not self.live:
            lt = self._trades.get(int(ticket))
            if lt:
                lt.closed = True; lt.close_reason = reason; lt.close_price = price
            print(f"    [DRY-RUN] CLOSE ticket={ticket} @ {price:.{digits}f} ({reason})")
            return True
        for fill_mode in self._get_filling_modes():
            bid, ask = self.quote()
            price = bid if ptype == 0 else ask
            req = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": round(vol, 2),
                "type": close_type,
                "position": int(ticket),
                "price": price,
                "deviation": deviation,
                "magic": MAGIC,
                "comment": reason[:31],
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": fill_mode,
            }
            res = _to_dict(self.mt5.order_send(req))
            retcode = int(res.get("retcode", -1))
            if retcode in DONE:
                return True
            if retcode in RETRY_RETCODES:
                if retcode == -1:
                    time.sleep(0.2)
                continue
            print(f"    [CLOSE-REJECT] ticket={ticket} retcode={retcode} {res.get('comment','')} fill={fill_mode}")
            return False
        return False

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

    def _deal_profit(self, ticket: int) -> float | None:
        """Try to pull realized profit (account ccy) from MT5 history deals for a closed position ticket."""
        try:
            from datetime import datetime as _dt
            from datetime import timedelta as _td
            # MT5's history_deals_select date range is interpreted in SERVER
            # time (the mt5linux bridge passes naive datetimes straight to the
            # Windows-side MT5 API). Exness MT5Trial9 runs at UTC+3, so the
            # v0.9.13.2 window of [now-1h, now+2min] in naive UTC excluded the
            # real deal — e.g. a close at 19:16 UTC = 22:16 server time was
            # outside 18:17-19:19 server, history_deals_get returned [], and
            # the bot fell back to the last-poll estimate (16:14: -9.60 booked
            # vs -37.75 real; 19:17: -3.68 booked vs -28.18 total). Use a
            # generous window that covers any server offset / DST: 7 days back
            # → now + 1 day.
            utc_to = _dt.utcnow() + _td(days=1)
            utc_from = utc_to - _td(days=7)
            self.mt5.history_deals_select(utc_from, utc_to)
            deals = self.mt5.history_deals_get(position=int(ticket)) or []
            total = 0.0
            for d in deals:
                dd = _to_dict(d)
                # entry deals have profit=0; closing/out deals carry the P&L
                p = float(dd.get("profit", 0.0))
                total += p
                # commission/swap also in their own fields on some brokers
                total += float(dd.get("commission", 0.0))
                total += float(dd.get("swap", 0.0))
            return total if deals else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Core signal handling
    # ------------------------------------------------------------------ #
    def _vlog(self, msg: str) -> None:
        if self.verbose:
            print(f"    [SIG] {msg}")

    def _vol_min_risk_ok(
        self,
        actual_risk_pct: float,
        target_risk_pct: float,
        side: str = "?",
        setup: str = "?",
        grade: str = "?",
        risk_per_unit: float = 0.0,
        equity: float = 0.0,
        *,
        silent: bool = False,
    ) -> bool:
        """Return True if floored size is within safe risk bounds (risk_cap only).

        v0.9.14 drops the 3×-target REJECT; hard rail is risk_cap only.
        """
        max_risk_cap = self.risk_cap
        cap_hit = actual_risk_pct > max_risk_cap
        if cap_hit:
            if not silent:
                why = f"risk={actual_risk_pct*100:.2f}% > cap={max_risk_cap*100:.2f}%"
                print(
                    f"    [REJECT] {side} {setup}/{grade} vol_min={self.spec.volume_min} forces "
                    f"{why} — SKIPPING "
                    f"(stop {risk_per_unit:.2f}pt too wide for min-lot)"
                )
            return False
        return True

    def _enforce_min_sl(self, entry: float, sl: float, direction: int,
                       atr: float, bid: float, ask: float) -> float:
        """Ensure SL is at least broker stop_level + safety margin from market.

        MT5 rejects orders with retcode 10016 ("Invalid stops") when SL/TP
        is closer than trade_stops_level points to the current price.  This
        method bumps the SL outward if needed, logging a warning.

        The minimum distance is: max(stop_level_pts * point, point * 500, 0.75 * ATR).
        The point*500 fallback handles brokers that report stop_level=0 but
        still reject tight stops (common on Exness).  0.75*ATR scales to
        any asset's volatility — BTC ATR≈19 → 14pt min; XAU ATR≈2 → 1.5pt min.
        v0.9.15.1: also enforces minimum TP distance (prevents 10016 on TP side).
        """
        point = self.spec.point
        broker_min = self._stop_level_pts * point
        fallback_min = point * 500
        min_dist = max(broker_min, fallback_min)
        if atr > 0:
            min_dist = max(min_dist, 0.75 * atr)
        market = ask if direction == 1 else bid
        if direction == 1:
            dist = market - sl
        else:
            dist = sl - market
        if dist < min_dist:
            if direction == 1:
                new_sl = market - min_dist
            else:
                new_sl = market + min_dist
            print(f"    [SL-BUMP] SL moved from {sl:.{self.spec.digits}f} to "
                  f"{new_sl:.{self.spec.digits}f} (min_dist={min_dist:.{self.spec.digits}f} "
                  f"broker_stop_level={self._stop_level_pts}pts)")
            return new_sl
        return sl

    @staticmethod
    def _parse_broker_time(raw: Any, fallback: datetime | None = None) -> datetime:
        """Coerce MT5 position.time (unix int / datetime / str) to aware UTC.

        MT5 returns position.time as a unix-seconds int. Try that first —
        ``pd.Timestamp(int)`` treats bare ints as *nanoseconds* which maps
        epoch-2026 seconds into 1970 and poisons orphan age / time-stop.
        """
        fb = fallback or datetime.now(UTC)
        if raw is None:
            return fb
        if isinstance(raw, datetime):
            return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
        # Numeric unix seconds (MT5 default). Reject absurd values so a
        # stray ms/ns epoch doesn't seed bars_held in the millions.
        try:
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                sec = float(raw)
                # allow ms timestamps too (13-digit) by scaling down
                if sec > 1e12:  # ms
                    sec /= 1000.0
                if sec > 1e12:  # still ns-ish — bail
                    return fb
                # sanity: between 2000-01-01 and 2100-01-01
                if 946684800.0 <= sec <= 4102444800.0:
                    return datetime.fromtimestamp(sec, tz=UTC)
        except Exception:
            pass
        try:
            ts = pd.Timestamp(raw)
            if pd.isna(ts):
                return fb
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.to_pydatetime()
        except Exception:
            return fb

    @staticmethod
    def _orphan_bars_held(open_time: datetime, now: datetime | None = None) -> int:
        """Seed bars_held from wall-clock age so time-stop doesn't reset on restart.

        A position open for 3h before a bot restart must still time-stop after
        ~1 more hour (240 M1 bars total), not get a fresh 4h clock.
        """
        now = now or datetime.now(UTC)
        ot = open_time if open_time.tzinfo is not None else open_time.replace(tzinfo=UTC)
        age_s = max(0.0, (now - ot).total_seconds())
        return int(age_s // 60)  # 1 M1 bar ≈ 60s

    def _adopt_orphan_position(self, ticket: int, pos: dict) -> LiveTrade | None:
        """Book a pre-existing magic-260810 position so CHoCH/time-stop apply."""
        if ticket in self._trades:
            return None
        ptype = int(pos.get("type", 0))
        direction = 1 if ptype == 0 else -1
        op = float(pos.get("price_open", 0.0))
        sl_p = float(pos.get("sl", 0.0))
        tp_p = float(pos.get("tp", 0.0))
        vol = float(pos.get("volume", self.spec.volume_min))
        ot = self._parse_broker_time(pos.get("time"), datetime.now(UTC))
        bars_held = self._orphan_bars_held(ot)
        # best-effort risk_pct (orphan; we don't know original grade)
        r_pct_est = 0.005
        lt = LiveTrade(
            ticket=ticket, direction=direction, entry=op, sl=sl_p,
            tp=tp_p, lots=vol, open_time=ot, grade="?",
            risk_pct=r_pct_est, bars_held=bars_held,
        )
        self._trades[ticket] = lt
        print(
            f"  [adopt] orphan ticket={ticket} {'LONG' if direction == 1 else 'SHORT'} "
            f"{vol} lots @ {op:.{self.spec.digits}f} SL={sl_p:.{self.spec.digits}f} "
            f"TP={tp_p:.{self.spec.digits}f} age≈{bars_held}m"
        )
        return lt

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
        # Single source of truth for open count: _our_open_positions() queries
        # MT5 directly for all positions with our magic number. v0.9.7 double-
        # counted by adding len(_trades) on top — _trades tracks LiveTrade
        # metadata for positions we opened THIS run, all of which are already
        # in the MT5 query. Orphan positions from prior bot runs that are not
        # in _trades ARE still in the MT5 query, so this correctly caps total
        # exposure (orphans count against max_open until they close broker-side).
        open_n = len(self._our_open_positions())
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
        risk_lots = self.spec.lots_for_risk(risk_per_unit, risk_quote)

        # Dynamic working-lot sizing:
        # working_lot is the USER's intended base trade size — treat it as
        # the absolute MINIMUM lot floor.  Don't scale it down by risk_pct
        # (that was the v0.9.15 bug: C-grade signals with 0.12% risk_pct
        # scaled 0.04 working_lot → 0.001, floored to vol_min 0.01, making
        # every trade a meaningless dust position).  risk_cap remains the
        # hard UPPER ceiling that prevents oversizing on wide stops.
        dynamic_target = float(np.clip(self.working_lot, self.spec.volume_min, self.spec.volume_max))
        dynamic_target = np.floor(dynamic_target / self.spec.volume_step) * self.spec.volume_step
        dynamic_target = float(dynamic_target)

        lots = max(risk_lots, dynamic_target) if risk_lots >= self.spec.volume_min else dynamic_target
        lots = np.floor(lots / self.spec.volume_step) * self.spec.volume_step
        lots = float(np.clip(lots, self.spec.volume_min, self.spec.volume_max))
        # Compute the *actual* risk the broker will enforce for the chosen lots
        # (vol_min floors lots, which can oversize risk on tiny accounts/grades).
        actual_risk_quote = self.spec.profit_per_lot(risk_per_unit) * lots
        actual_risk_acct = self.acct.to_account_ccy(actual_risk_quote, self.spec.currency_profit)
        actual_risk_pct = actual_risk_acct / max(equity, 1e-9)
        if lots < self.spec.volume_min - 1e-9:
            self._vlog(f"{side} {setup}/{sig.grade} rejected: lots={lots:.4f} < vol_min={self.spec.volume_min} (eq={equity:.0f} risk_pct={risk_pct:.4f})")
            return
        # vol_min floor safety: if min-lot forces ACTUAL risk > 1.5% (or 3x the
        # target risk_pct), REJECT the trade. C-grade LIQ_SWEEP/BOS_CONT scalps
        # with wide stops (5+ pts) get floored to 0.01 lots which can push real
        # risk to 3%+ — that's 20x the intended size and violates the scalping
        # ethos (quick 0.25-0.5% nibbles, not 3% gambles). Don't let min-lots
        # turn scalps into swing bets. v0.9.11/v0.9.12 only SIZE-WARNed;
        # v0.9.13 hard-rejects.
        if not self._vol_min_risk_ok(actual_risk_pct, risk_pct, side, setup, sig.grade, risk_per_unit, equity):
            return
        if actual_risk_pct > risk_pct * 1.25:
            # mild oversizing within bounds — warn but proceed
            print(f"    [SIZE-WARN] {side} {setup}/{sig.grade} vol_min={self.spec.volume_min} forced "
                  f"risk={actual_risk_pct*100:.2f}% (target {risk_pct*100:.2f}%) eq={equity:.0f}")
            risk_pct = actual_risk_pct
        tp = fill + sig.direction * self.cfg.exits.tp1_r * risk_per_unit
        sl = float(sig.stop)
        # v0.9.15.1: enforce broker minimum stop distance (prevents 10016 rejects)
        sl = self._enforce_min_sl(fill, sl, sig.direction, sig.atr_at_entry, bid, ask)
        risk_per_unit = abs(fill - sl)
        if risk_per_unit <= 0:
            self._vlog(f"{side} {setup}/{sig.grade} rejected: invalid risk_per_unit after SL bump")
            return
        tp = fill + sig.direction * self.cfg.exits.tp1_r * risk_per_unit
        # v0.9.15.1: enforce minimum TP distance too — broker rejects TP too close
        point = self.spec.point
        tp_market_dist = abs(tp - (ask if sig.direction == 1 else bid))
        tp_min = max(point * 500, 0.75 * sig.atr_at_entry) if sig.atr_at_entry > 0 else point * 500
        if tp_market_dist < tp_min:
            tp = (ask if sig.direction == 1 else bid) + sig.direction * tp_min
            print(f"    [TP-BUMP] TP moved to {tp:.{self.spec.digits}f} (min_dist={tp_min:.{self.spec.digits}f})")
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
        # v0.9.15: compute TP2 price for hybrid ladder
        tp2_price = fill + sig.direction * self.cfg.exits.tp2_r * risk_per_unit
        self._trades[ticket] = LiveTrade(
            ticket=ticket, direction=sig.direction, entry=fill, sl=sl, tp=tp,
            lots=lots, open_time=datetime.now(UTC), grade=sig.grade, risk_pct=risk_pct,
            tp2_price=tp2_price, remaining_lots=lots, original_lots=lots,
        )
        # One-shot arm for BOS_CONT: after a successful fill, mark this leg
        # as "entered" so subsequent consecutive BOS bars in the same leg
        # don't pyramid additional positions. Reset by opposite CHoCH in
        # Phase 1b of _evaluate_row (signals.py). This is set HERE in the
        # live trader (and in the backtest engine), NOT inside _evaluate_row,
        # so state-priming warmups don't prematurely arm the key.
        if setup == "BOS_CONT":
            self._state[f"_bos_entered_{sig.direction:+d}"] = True
        print(f"    [ENTRY] ticket={ticket} {side} {setup} {lots} lots @ {fill:.{self.spec.digits}f} "
              f"grade={sig.grade} {zone_label} kz={sig.killzone} SL={sl:.{self.spec.digits}f} TP={tp:.{self.spec.digits}f}")

    # ------------------------------------------------------------------ #
    # Position monitoring
    # ------------------------------------------------------------------ #
    def _monitor_positions(self, latest_m1_row: pd.Series | None, *, new_bar: bool = False) -> None:
        """Check SL/TP/emergency/time-stop for open positions.

        v0.9.15 hybrid ladder: TP1 1.0R @ 50% → BE; TP2 2.5R @ 25%;
        runner 25% with ATR trail + M5 CHoCH kill.

        new_bar=True means this call corresponds to a fresh closed M1 bar —
        that's where bars_held increments and CHoCH/time-stop checks run.
        Intra-bar poll calls (new_bar=False) only check broker-side SL/TP.
        """
        bid, ask = self.quote()
        atr = 0.0
        if latest_m1_row is not None:
            atr = float(latest_m1_row.get('atr_14', 0.0)) if pd.notna(latest_m1_row.get('atr_14')) else 0.0

        # ---------- SL / TP (hybrid ladder) ---------- #
        if not self.live:
            to_close: list[tuple[int, str]] = []
            for ticket, lt in list(self._trades.items()):
                if lt.closed:
                    continue
                price = bid if lt.direction == 1 else ask
                # Check SL hit
                hit_sl = (lt.direction == 1 and price <= lt.sl) or (lt.direction == -1 and price >= lt.sl)
                if hit_sl:
                    to_close.append((ticket, "SL"))
                    continue
                # Hybrid ladder: TP1 → partial close + BE
                if not lt.tp1_hit:
                    hit_tp1 = (lt.direction == 1 and price >= lt.tp) or (lt.direction == -1 and price <= lt.tp)
                    if hit_tp1:
                        lt.tp1_hit = True
                        # Close 50% at TP1
                        close_lots = lt.original_lots * self.cfg.exits.tp1_pct
                        close_lots = max(round(close_lots, 2), self.spec.volume_min)
                        lt.remaining_lots = max(round(lt.original_lots - close_lots, 2), self.spec.volume_min)
                        # Move SL to breakeven
                        lt.sl = lt.entry
                        print(f"    [TP1] ticket={ticket} closed {close_lots} lots @ {price:.{self.spec.digits}f} "
                              f"→ BE SL={lt.sl:.{self.spec.digits}f} remaining={lt.remaining_lots}")
                # TP2 → partial close + trail
                elif not lt.tp2_hit:
                    hit_tp2 = (lt.direction == 1 and price >= lt.tp2_price) or (lt.direction == -1 and price <= lt.tp2_price)
                    if hit_tp2:
                        lt.tp2_hit = True
                        close_lots = lt.original_lots * self.cfg.exits.tp2_pct
                        close_lots = max(round(close_lots, 2), self.spec.volume_min)
                        lt.remaining_lots = max(round(lt.remaining_lots - close_lots, 2), self.spec.volume_min)
                        print(f"    [TP2] ticket={ticket} closed {close_lots} lots @ {price:.{self.spec.digits}f} "
                              f"remaining={lt.remaining_lots}")
                # Runner: ATR trail + M5 CHoCH kill
                else:
                    # M5 CHoCH kill for runner
                    if latest_m1_row is not None:
                        m5_choch_against = (
                            (lt.direction == 1 and bool(latest_m1_row.get("M5_minor_choch_dn", False))) or
                            (lt.direction == -1 and bool(latest_m1_row.get("M5_minor_choch_up", False)))
                        )
                        if m5_choch_against:
                            to_close.append((ticket, "M5_CHOCH_RUNNER"))
                            continue
                    # ATR trailing stop for runner
                    if atr > 0:
                        trail_dist = self.cfg.exits.runner_trail_atr_mult * atr
                        if lt.direction == 1:
                            new_trail = price - trail_dist
                            if new_trail > lt.sl:
                                lt.sl = new_trail
                        else:
                            new_trail = price + trail_dist
                            if new_trail < lt.sl:
                                lt.sl = new_trail
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
                # v0.9.15: one-shot re-arm when flat after TP
                if reason in ("TP1", "M5_CHOCH_RUNNER"):
                    # Re-arm BOS_CONT one-shot so next leg can fire
                    self._state[f"_bos_entered_{lt.direction:+d}"] = False
            broker_pos: dict[int, dict] = {}
        else:
            broker_pos = self._our_open_positions()
            broker_tickets = set(broker_pos.keys())
            prev = getattr(self, "_last_broker_pos", {}) or {}
            for ticket, lt in list(self._trades.items()):
                if ticket not in broker_tickets and not lt.closed:
                    realized = self._deal_profit(int(ticket))
                    if realized is not None:
                        last_profit = realized
                    else:
                        last_profit = float(prev.get(ticket, {}).get("profit", 0.0)) if prev else 0.0
                        print(f"    [WARN] ticket={ticket} no deal in MT5 history — using last-poll estimate "
                              f"{last_profit:+.2f}{self.acct.currency}")
                    lt.closed = True
                    lt.close_reason = "BROKER"
                    lt.pnl = last_profit
                    if last_profit > 0:
                        self._wins_count += 1
                    print(f"    [EXIT] ticket={ticket} reason=BROKER (SL/TP hit) pnl={last_profit:+.2f}{self.acct.currency}")
                    # v0.9.15: one-shot re-arm when flat after TP
                    self._state[f"_bos_entered_{lt.direction:+d}"] = False
            self._last_broker_pos = dict(broker_pos)

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
            # Adopt any existing magic-260810 positions already open on the
            # broker (orphans from prior bot runs / restarts) so CHoCH/
            # time-stop protection applies and P&L bookkeeping tracks them.
            # Without this, an open scalp from the previous run would sit
            # unmonitored until broker SL/TP — no emergency exit, no time-stop.
            # bars_held is seeded from wall-clock age so a 3h-old orphan still
            # time-stops after ~1 more hour (not a fresh 4h clock on restart).
            for ticket, pos in self._our_open_positions().items():
                self._adopt_orphan_position(ticket, pos)
            self._last_broker_pos = self._our_open_positions()
            print(f"  Warmed up on {n} M1 bars — state primed, watching for new signals ...")

        # Only process bars newer than the last one we already evaluated.
        new_mask = pd.Series(True, index=aligned.index)
        if self._last_processed_m1_time is not None:
            new_mask = aligned["time"] > self._last_processed_m1_time
        new_rows = aligned.index[new_mask].tolist()

        n_sigs_this_cycle = 0
        verbose_rejects: list[str] = []
        for i in new_rows:
            row = aligned.iloc[i]
            trace: list[str] = [] if self.verbose else []
            try:
                sig = _evaluate_row(int(i), row, self.cfg, self._state, fail_trace=trace)
            except Exception as e:
                sig = None
                if self.verbose:
                    trace.append(f"[row {i}] EXCEPTION: {e}")
            if sig is not None:
                self._handle_signal(sig)
                n_sigs_this_cycle += 1
            elif self.verbose and trace:
                # In verbose mode log reject reasons for 'interesting' bars
                # (disp/BOS/sweep just fired) so we can see which gate is
                # killing the row; also log once per 5 cycles unconditionally.
                # Print [GATE] reason for any bar carrying a structural
                # impulse flag on M1 or the trigger TF, so verbose output
                # tells us exactly which gate killed a "hot" bar.
                flags_hot = False
                for fl in ('bear_disp', 'bull_disp',
                           'minor_bos_up', 'minor_bos_dn',
                           'minor_choch_up', 'minor_choch_dn',
                           'major_bos_up', 'major_bos_dn',
                           'major_choch_up', 'major_choch_dn',
                           'bull_liq_sweep', 'bear_liq_sweep',
                           'M5_bull_disp', 'M5_bear_disp',
                           'M5_minor_bos_up', 'M5_minor_bos_dn',
                           'M5_minor_choch_up', 'M5_minor_choch_dn'):
                    if bool(row.get(fl, False)):
                        flags_hot = True
                        break
                if flags_hot or self._cycle % 5 == 0:
                    verbose_rejects.append(trace[-1])

        if self.verbose and verbose_rejects:
            for line in verbose_rejects[:8]:
                print(f"    [GATE] {line}")
            if len(verbose_rejects) > 8:
                print(f"    [GATE] ... and {len(verbose_rejects)-8} more rejects this cycle")

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
        # wins count from per-trade reconciled pnl (snapshot-based in live,
        # r_mult-based in dry-run)
        wins = sum(1 for t in closed if t.pnl > 0)
        total_pnl = sum(t.pnl for t in closed) if closed else 0.0
        # ground-truth realized P&L from broker balance (includes trades
        # from before bot restart + commissions)
        bal = float(acc.get("balance", eq))
        if self._starting_balance is None:
            self._starting_balance = bal
        realized_gt = bal - self._starting_balance
        floating = 0.0
        if self.live:
            # MT5 position.profit is ALREADY in account (deposit) currency
            # on Exness ZAR accounts -- broker does USD->ZAR server-side.
            # Multiplying by fx again (v0.9.9 bug) 18.5x'd the display.
            for p in self._our_open_positions().values():
                floating += float(p.get("profit", 0.0))
        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        bar_time = pd.to_datetime(latest["time"]).strftime("%H:%M") if "time" in latest.index else "?"
        print(
            f"[{now_str}] M1 {bar_time}  bid={bid:.{self.spec.digits}f} ask={ask:.{self.spec.digits}f}  "
            f"eq={eq:.2f}{acc.get('currency','ZAR')}  open={n_open}  floating={floating:+.2f}  "
            f"closed={len(closed)} wins={wins} realized_session={total_pnl:+.2f} "
            f"realized_total={realized_gt:+.2f}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Sleep hardening
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clamp_sleep(seconds: float, *, max_seconds: float | None = None) -> float:
        """Clamp a computed sleep duration so ``time.sleep`` NEVER gets a negative value.

        Python raises ``ValueError: sleep length must be non-negative`` for
        negative durations, and an uncaught ValueError inside the run loop
        kills the whole trader. The bar-boundary poller races the wall clock::

            while running["v"] and time.time() < end_wait:
                ...monitor...
                time.sleep(min(self.poll_interval, end_wait - time.time()))

        If the monitor call (or scheduler jitter) crosses ``end_wait`` between
        the loop check and the sleep computation, ``end_wait - time.time()``
        goes negative and ``min()`` hands a negative value straight to
        ``time.sleep`` — that is the 13:14 crash. v0.9.13.1+ routes every
        ``time.sleep`` through this guard.

        Returns a safe, non-negative duration, optionally capped at
        ``max_seconds`` (cap itself is clamped to >= 0). NaN / +/-inf /
        non-numeric garbage collapse to 0.0 so erratic inputs can never crash
        the loop either.
        """
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return 0.0
        if seconds != seconds or seconds in (float("inf"), float("-inf")):
            seconds = 0.0
        if max_seconds is not None:
            try:
                cap = float(max_seconds)
            except (TypeError, ValueError):
                cap = 0.0
            if cap != cap:  # NaN
                cap = 0.0
            # cap=+inf means "no cap"; cap=-inf clamps to 0
            seconds = min(seconds, max(0.0, cap))
        return max(0.0, seconds)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        persona_label = "v0.9.15 SCALPER (all setups, RL-unrestricted)" if not self.cfg.confluence.persona_gating else "v0.9.15 champion (long-only A+/A/B hybrid ladder)"
        print(f"SlyTrade LIVE v0.9.15  symbol={self.symbol}  live={self.live}  risk_cap={self.risk_cap*100:.1f}%  working_lot={self.working_lot}  max_open={self.max_open}")
        print(f"persona: {persona_label}")
        print("setups : RETEST_OB RETEST_FVG LIQ_SWEEP BOS_CONT DISP_TRAP BREAKER  (champion gates apply unless --all)")
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
                # v0.9.13.1: clamp BEFORE sleep — a negative remaining
                # duration (clock crossing end_wait mid-iteration) made
                # time.sleep raise ValueError and killed the loop at 13:14.
                time.sleep(self._clamp_sleep(end_wait - time.time(), max_seconds=self.poll_interval))
            if not running["v"]: break
            try:
                self._cycle_fn()
            except Exception as e:
                print(f"[ERROR] cycle: {e}")
                traceback.print_exc()
                time.sleep(self._clamp_sleep(10.0))


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="SlyTrade v0.9.15 SCALPER LIVE trader (hybrid ladder + DISP_TRAP/BREAKER + SL clamp + limit retests)")
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--live", action="store_true", help="Actually send orders (default: dry-run)")
    ap.add_argument("--risk-cap", type=float, default=0.01, help="Max risk per trade as fraction of equity (default 1%%)")
    ap.add_argument("--working-lot", type=float, default=0.04, help="Working lot size for dynamic sizing (default 0.04)")
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

    resolved, spec, stop_level_pts = resolve_symbol_spec(mt5, args.symbol, str(acc.get("currency","ZAR")), args.usd_zar)
    print(f"  symbol resolved: {resolved} (point={spec.point} digits={spec.digits} "
          f"contract={spec.contract_size} vol_min={spec.volume_min} vol_step={spec.volume_step} "
          f"stop_level={stop_level_pts}pts)")

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
    print(f"  working_lot   : {args.working_lot}")
    print(f"  max_open      : {max_open_eff}")
    trader = LiveTrader(
        mt5=mt5, symbol=resolved, spec=spec, cfg=cfg, acct=acct_spec,
        live=args.live, risk_cap=args.risk_cap, working_lot=args.working_lot, max_open=max_open_eff,
        verbose=args.verbose, stop_level_pts=stop_level_pts,
    )
    try:
        trader.run()
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
