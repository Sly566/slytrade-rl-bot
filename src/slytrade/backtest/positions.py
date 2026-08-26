"""Layer 5 — position model for hedging mode.

Each ICT setup creates a Position with three tranches:
  - T1: 40% size → TP1 (1R), then SL moves to entry (BE) for remaining
  - T2: 30% size → TP2 (2R), then runner SL moves to TP1
  - T3: 30% size → runner (HTF swing target / 3R fallback), trailed by
       0.5× ATR14 after TP2, killed on M5 CHoCH against, killed on
       M15 CHoCH against (which also force-exits T1/T2 if still open)
  - Time stop: 240 M1 bars (4h) after entry, close any remaining at market
  - Emergency: M15 CHoCH against → full exit at next bar open

Each tranche tracks its own entry, SL, TP, state (open/filled/closed), exit
price, exit reason, and P&L.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# --------------------------------------------------------------------------- #
# State enums
# --------------------------------------------------------------------------- #
class TrancheState(str):
    OPEN = "open"         # SL/TP/trail live
    CLOSED = "closed"     # exited at a price
    CANCELED = "canceled" # never filled (e.g. T1 closed but runner trail didn't trigger)


class ExitReason(str):
    SL = "sl"                       # initial stop hit
    TP1 = "tp1"                     # TP1 hit (T1 exit)
    TP2 = "tp2"                     # TP2 hit (T2 exit)
    BE = "be"                       # SL moved to entry hit after TP1 (T2+T3 flat)
    TRAIL = "trail"                 # ATR trailing stop hit (T3 runner)
    M5_CHOCH = "m5_choch"           # M5 CHoCH against (runner exit)
    M15_CHOCH = "m15_choch"         # M15 CHoCH against (emergency full exit)
    TIME_STOP = "time_stop"         # 240 bars elapsed, no progress
    RUNNER_TARGET = "runner_target" # T3 runner reached final swing target
    END_OF_DATA = "end_of_data"     # backtest ended while still open


class Direction(int):
    LONG = 1
    SHORT = -1


@dataclass
class Tranche:
    name: str                  # "T1" / "T2" / "T3"
    size_frac: float           # fraction of total position lots
    lots: float                # actual lots
    entry: float               # entry price (same for all tranches)
    sl: float                  # current stop price
    tp: float | None        # current target price (None for runner when trailing)
    state: str = TrancheState.OPEN
    exit_price: float | None = None
    exit_reason: str | None = None
    exit_time: pd.Timestamp | None = None
    exit_bars: int = 0

    def pnl_ccy(self, price: float, direction: int, spec) -> float:
        """Mark-to-market P&L in quote currency for this tranche at `price`.

        direction=+1 (long): profit when price > entry.
        direction=-1 (short): profit when price < entry (i.e. entry - exit).
        """
        if not self.entry:
            return 0.0
        raw = price - self.entry
        signed = raw * direction  # long: +raw, short: -raw = entry - price
        return spec.profit_per_lot(signed) * self.lots

    def realized_pnl_ccy(self, direction: int, spec) -> float:
        if self.state != TrancheState.CLOSED or self.exit_price is None:
            return 0.0
        signed = (self.exit_price - self.entry) * direction
        return spec.profit_per_lot(signed) * self.lots


@dataclass
class Position:
    # Core
    pos_id: int
    symbol: str
    direction: int                 # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float             # fill price (after spread)
    total_lots: float
    atr_at_entry: float
    grade: str
    risk_pct: float                # fraction of equity risked at entry
    risk_per_unit_quote: float     # |entry - initial_sl| in quote ccy (USD for XAU)
    # Targets
    initial_sl: float
    tp1: float
    tp2: float
    tp_runner: float
    swing_target_tf: str
    swing_target_price: float
    # Confluence / context
    trigger_tf: str
    ob_tf: str | None
    zone_kind: str                 # "OB" / "FVG"
    killzone: str
    session: str
    confluence_tags: list[str] = field(default_factory=list)
    htf_bias_summary: dict[str, int] = field(default_factory=dict)

    # Tranches
    tranches: list[Tranche] = field(default_factory=list)

    # State flags
    tp1_hit: bool = False
    tp2_hit: bool = False
    be_lock: bool = False          # after TP1: SL moved to entry for T2+T3
    runner_trailing: bool = False  # after TP2: SL trails by ATR for T3
    trail_sl: float | None = None
    bars_held: int = 0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Exit metadata (filled when fully closed)
    close_time: pd.Timestamp | None = None
    close_reason: str | None = None

    # Internal
    _entry_cost_paid: bool = False

    # ------------------------------------------------------------------ #
    def init_tranches(self, volume_min: float = 0.01, volume_step: float = 0.01,
                      t1_frac: float = 1.0, t2_frac: float = 0.0, t3_frac: float = 0.0) -> None:
        """Allocate lots across T1/T2/T3 respecting broker volume granularity.

        Default: single tranche (100% @ TP1) for the battle-tested 0.75R scalp.
        Pass t1_frac/t2_frac/t3_frac for laddered exits.
        """
        def snap(x: float) -> float:
            return round(int(x / volume_step) * volume_step, 4) if x > 0 else 0.0

        fracs = [("T1", t1_frac, self.tp1),
                 ("T2", t2_frac, self.tp2),
                 ("T3", t3_frac, self.tp_runner)]
        # Drop zero fractions
        active = [(n, f, t) for n, f, t in fracs if f > 0.001]
        if not active:
            # Fallback: single T1
            active = [("T1", 1.0, self.tp1)]

        len(active)
        alloc: dict[str, float] = {}
        remaining = self.total_lots
        # Snap all but last to volume_step
        for i, (name, frac, _tp) in enumerate(active):
            if i < len(active) - 1:
                lots = snap(self.total_lots * frac)
                # Ensure at least volume_min if we allocated >= volume_min in raw terms
                if self.total_lots * frac >= volume_min * 0.5:
                    lots = max(lots, volume_min)
                lots = min(lots, max(0.0, round(remaining - (len(active)-i-1)*volume_min, 4)))
                alloc[name] = lots
                remaining = round(remaining - lots, 4)
            else:
                # Last tranche takes the remainder (ensures totals match)
                alloc[name] = max(0.0, round(remaining, 4))

        for name, frac, tp in active:
            lots = alloc.get(name, 0.0)
            if lots >= volume_min - 1e-9:
                self.tranches.append(Tranche(
                    name, frac, lots,
                    self.entry_price, self.initial_sl, tp))

        # Safety: if no tranches were created (e.g., all rounded to 0), single T1
        if not self.tranches:
            self.tranches.append(Tranche(
                "T1", 1.0, round(self.total_lots, 4),
                self.entry_price, self.initial_sl, self.tp1))

    # ------------------------------------------------------------------ #
    def open_tranches(self) -> list[Tranche]:
        return [t for t in self.tranches if t.state == TrancheState.OPEN]

    def is_closed(self) -> bool:
        return len(self.open_tranches()) == 0

    def update_excursion(self, high: float, low: float) -> None:
        if self.direction == 1:
            mfe = high - self.entry_price
            mae = self.entry_price - low
        else:
            mfe = self.entry_price - low
            mae = high - self.entry_price
        self.max_favorable_excursion = max(self.max_favorable_excursion, mfe)
        self.max_adverse_excursion = max(self.max_adverse_excursion, mae)

    # ------------------------------------------------------------------ #
    def close_tranche(self, name: str, price: float, reason: str,
                      time: pd.Timestamp) -> Tranche | None:
        for t in self.tranches:
            if t.name == name and t.state == TrancheState.OPEN:
                t.exit_price = price
                t.exit_reason = reason
                t.exit_time = time
                t.exit_bars = self.bars_held
                t.state = TrancheState.CLOSED
                return t
        return None

    def close_all(self, price: float, reason: str, time: pd.Timestamp) -> None:
        for t in self.open_tranches():
            t.exit_price = price
            t.exit_reason = reason
            t.exit_time = time
            t.exit_bars = self.bars_held
            t.state = TrancheState.CLOSED
        self.close_time = time
        self.close_reason = reason

    # ------------------------------------------------------------------ #
    def total_realized_pnl(self, spec) -> float:
        return sum(t.realized_pnl_ccy(self.direction, spec) for t in self.tranches)

    def unrealized_pnl(self, price: float, spec) -> float:
        return sum(t.pnl_ccy(price, self.direction, spec) for t in self.open_tranches())

    def r_multiple_realized(self, spec) -> float:
        """Realized P&L expressed as R-multiples (R = full initial risk)."""
        r_per_lot = spec.profit_per_lot(self.risk_per_unit_quote) * self.total_lots
        if r_per_lot <= 0:
            return 0.0
        return self.total_realized_pnl(spec) / r_per_lot

    def r_multiple_total(self, price: float, spec) -> float:
        r_per_lot = spec.profit_per_lot(self.risk_per_unit_quote) * self.total_lots
        if r_per_lot <= 0:
            return 0.0
        return (self.total_realized_pnl(spec) + self.unrealized_pnl(price, spec)) / r_per_lot
