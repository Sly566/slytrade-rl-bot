"""Layer 6-ready LIVE trading loop for SlyTrade v0.9.0 champion persona.

Connects to MT5 via the mt5linux RPyC bridge (run `bash start_mt5_bridge.sh`
in another terminal first), pulls multi-timeframe bars, computes Layer 2
features, performs causal MTF alignment, runs the Layer 4/5 signal scanner
statefully, and when a signal fires sends a market order with SL/TP sized
per our grade-tiered risk rules.

Open positions are monitored every 5 seconds for:
  - SL hit (broker-side, redundant check via equity diff)
  - TP hit (broker-side)
  - M15 CHoCH emergency exit (closes at market if detected)
  - Time-stop (4 hours / 240 M1 bars flat if <0R)

Run:
    bash start_mt5_bridge.sh        # in another terminal
    python -m slytrade.live.trader --symbol XAUUSDm --dry-run   # first, test
    python -m slytrade.live.trader --symbol XAUUSDm --live      # real trading

Use --dry-run to print orders without sending. Default risk: champion
persona tiered sizing (A+=1%, A=0.75%, B=0.5%). Override with --risk-cap 0.01
for 1% max risk on the demo.
"""
from __future__ import annotations

import argparse
import signal
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# Imports from our own Layer 2-5 stack
# --------------------------------------------------------------------------- #
from ..backtest.specs import AccountSpec, SymbolSpec, spec_for_symbol
from ..data.features import DEFAULT_CONFIG, process_bars
from ..data.mtf_align import _asof_merge, _prep_htf_frame
from ..data.schemas import normalize_bar_frame
from ..data.time import timeframe_timedelta
from ..strategy.config import StrategyConfig, champion_persona
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
WARMUP_BARS = {"M1": 600, "M5": 400, "M15": 300, "H1": 200, "H4": 120, "D1": 60}
HTFS = ["M5", "M15", "H1", "H4", "D1"]
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
    """Fetch `count` bars ending 'now' from MT5, normalized to our schema."""
    tf_const = getattr(mt5, TIMEFRAME_ATTRS[timeframe])
    raw = mt5.copy_rates_from_pos(symbol, tf_const, 0, int(count))
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    return normalize_bar_frame(raw, symbol, timeframe)


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

        self._state: dict = {}
        self._trades: dict[int, LiveTrade] = {}
        self._last_processed_m1_time: datetime | None = None
        self._signals_fired: set[str] = set()   # dedupe keys
        self._cycle = 0

    # ------------------------------------------------------------------ #
    # Account info
    # ------------------------------------------------------------------ #
    def account(self) -> dict:
        return _to_dict(self.mt5.account_info())

    def equity(self) -> float:
        return float(self.account().get("equity", 0.0))

    def quote(self) -> tuple[float, float]:
        t = _to_dict(self.mt5.symbol_info_tick(self.symbol))
        return float(t.get("bid", 0.0)), float(t.get("ask", 0.0))

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
    def _handle_signal(self, sig) -> None:
        key = f"{sig.time.isoformat()}|{sig.direction}|{sig.grade}|{sig.ob_tf or ''}"
        if key in self._signals_fired:
            return
        # Skip shorts in champion persona
        if sig.direction == -1 and not self.cfg.confluence.accept_shorts:
            self._signals_fired.add(key); return
        if sig.direction == 1 and not self.cfg.confluence.accept_longs:
            self._signals_fired.add(key); return
        open_n = len(self._our_open_positions()) + sum(1 for t in self._trades.values() if not t.closed)
        if open_n >= self.max_open:
            return
        equity = self.equity()
        bid, ask = self.quote()
        entry_approx = ask if sig.direction == 1 else bid
        half_spread = (ask - bid) * 0.5
        slip_pts = 5 * self.spec.point
        fill = entry_approx + (half_spread + slip_pts) if sig.direction == 1 else entry_approx - (half_spread + slip_pts)
        risk_per_unit = abs(fill - float(sig.stop))
        if risk_per_unit <= 0:
            return
        risk_pct = min(float(sig.risk_pct), self.risk_cap)
        risk_acct = risk_pct * equity
        risk_quote = risk_acct / self.acct.fx_to_account.get(self.spec.currency_profit, 1.0)
        lots = self.spec.lots_for_risk(risk_per_unit, risk_quote)
        if lots < self.spec.volume_min - 1e-9:
            return
        tp = fill + sig.direction * self.cfg.exits.tp1_r * risk_per_unit
        sl = float(sig.stop)
        margin_quote = (lots * self.spec.contract_size * fill) / max(self.acct.leverage, 1)
        margin_acct = self.acct.to_account_ccy(margin_quote, self.spec.currency_profit)
        if margin_acct > equity * 0.95:
            return
        comment = f"L5 {sig.grade} {sig.ob_tf or ''} {sig.killzone}"
        ticket = self._place_market(sig.direction, lots, sl, tp, comment)
        if ticket is None:
            return
        self._signals_fired.add(key)
        self._trades[ticket] = LiveTrade(
            ticket=ticket, direction=sig.direction, entry=fill, sl=sl, tp=tp,
            lots=lots, open_time=datetime.now(UTC), grade=sig.grade, risk_pct=risk_pct,
        )
        print(f"    [ENTRY] ticket={ticket} {'LONG' if sig.direction==1 else 'SHORT'} {lots} lots @ {fill:.{self.spec.digits}f} "
              f"grade={sig.grade} ob={sig.ob_tf} kz={sig.killzone} SL={sl:.{self.spec.digits}f} TP={tp:.{self.spec.digits}f}")

    # ------------------------------------------------------------------ #
    # Position monitoring
    # ------------------------------------------------------------------ #
    def _monitor_positions(self, latest_m1_row: pd.Series | None) -> None:
        if self.live:
            broker_pos = self._our_open_positions()
        else:
            broker_pos = {}
        if not self.live:
            bid, ask = self.quote()
            to_close: list[tuple[int, str]] = []
            for ticket, lt in list(self._trades.items()):
                if lt.closed:
                    continue
                price = bid if lt.direction == 1 else ask
                if lt.direction == 1:
                    if price <= lt.sl: to_close.append((ticket, "SL"))
                    elif price >= lt.tp: to_close.append((ticket, "TP1"))
                else:
                    if price >= lt.sl: to_close.append((ticket, "SL"))
                    elif price <= lt.tp: to_close.append((ticket, "TP1"))
                lt.bars_held += 1
            for ticket, reason in to_close:
                if self._close_position(ticket, reason):
                    print(f"    [EXIT] ticket={ticket} reason={reason}")
            return
        # Live mode: reconcile broker positions
        broker_tickets = set(broker_pos.keys())
        for ticket, lt in list(self._trades.items()):
            if ticket not in broker_tickets and not lt.closed:
                lt.closed = True
                lt.close_reason = "BROKER"
                print(f"    [EXIT] ticket={ticket} reason=BROKER (SL/TP hit)")
        for ticket in broker_tickets:
            if ticket not in self._trades:
                continue
            lt = self._trades[ticket]
            lt.bars_held += 1
            # M15 CHoCH emergency
            if latest_m1_row is not None:
                if lt.direction == 1 and bool(latest_m1_row.get(f"{CHOPCH_EMERGENCY_TF}_major_choch_dn", False)):
                    print(f"    [EMERGENCY M15 CHoCH DOWN against LONG ticket={ticket} — closing")
                    self._close_position(ticket, "M15_CHOCH")
                elif lt.direction == -1 and bool(latest_m1_row.get(f"{CHOPCH_EMERGENCY_TF}_major_choch_up", False)):
                    print(f"    [EMERGENCY M15 CHoCH UP against SHORT ticket={ticket} — closing")
                    self._close_position(ticket, "M15_CHOCH")
            if lt.bars_held >= TIME_STOP_BARS and not lt.closed:
                pos = broker_pos[ticket]
                cur_profit = float(pos.get("profit", 0.0))
                if cur_profit < 0:
                    print(f"    [TIME-STOP] ticket={ticket} profit={cur_profit:.2f} — closing")
                    self._close_position(ticket, "TIME_STOP")

    # ------------------------------------------------------------------ #
    # One M1 cycle
    # ------------------------------------------------------------------ #
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
        new_mask = pd.Series(True, index=aligned.index)
        if self._last_processed_m1_time is not None:
            new_mask = aligned["time"] > self._last_processed_m1_time
        new_rows = aligned.index[new_mask].tolist()
        feed_start = max(0, min(new_rows[0] if new_rows else n, n - 200) - 50)
        for i in range(feed_start, n):
            row = aligned.iloc[i]
            try:
                sig = _evaluate_row(int(i), row, self.cfg, self._state)
            except Exception:
                sig = None
            if sig is not None and i >= (new_rows[0] if new_rows else n):
                self._handle_signal(sig)
        if new_rows:
            self._last_processed_m1_time = aligned.iloc[new_rows[-1]]["time"]
        latest = aligned.iloc[-1]
        self._monitor_positions(latest)
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
        print(f"SlyTrade LIVE v0.9.0  symbol={self.symbol}  live={self.live}  risk_cap={self.risk_cap*100:.1f}%  max_open={self.max_open}")
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
    ap = argparse.ArgumentParser(description="SlyTrade v0.9.0 LIVE trader")
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--live", action="store_true", help="Actually send orders (default: dry-run)")
    ap.add_argument("--risk-cap", type=float, default=0.01, help="Max risk per trade as fraction of equity (default 1%%)")
    ap.add_argument("--max-open", type=int, default=3)
    ap.add_argument("--usd-zar", type=float, default=18.5)
    ap.add_argument("--leverage", type=int, default=2000)
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
    cfg = champion_persona()
    trader = LiveTrader(
        mt5=mt5, symbol=resolved, spec=spec, cfg=cfg, acct=acct_spec,
        live=args.live, risk_cap=args.risk_cap, max_open=args.max_open,
    )
    try:
        trader.run()
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
