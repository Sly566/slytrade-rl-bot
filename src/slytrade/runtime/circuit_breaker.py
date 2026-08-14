"""Trading-limit circuit breaker.

Implements the discipline a professional ICT trader keeps manually:

* a hard cap on consecutive losing trades with a mandatory cooldown,
* a daily loss-count cap,
* a daily trade-count cap (prevents overtrading / revenge trading).

All limits are driven by ``configs/risk.yaml`` — nothing is hard-coded. The
breaker only ever *pauses new entries*; it never blocks risk-reducing exits,
which is the same invariant the guardrails already enforce.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta


@dataclass(frozen=True)
class TradingLimits:
    """Risk limits loaded from ``configs/risk.yaml`` (see ``limits_from_config``)."""

    max_consecutive_losses: int = 3
    max_daily_losses: int = 5
    max_daily_trades: int = 0  # 0 = unlimited
    cooldown_minutes: int = 30
    risk_per_trade: float = 0.005
    max_daily_drawdown: float = 0.03
    max_total_drawdown: float = 0.08


@dataclass(frozen=True)
class BreakerDecision:
    allowed: bool
    reason: str = "ok"


def limits_from_config(risk: dict) -> TradingLimits:
    """Read the circuit-breaker limits from the risk config mapping."""
    return TradingLimits(
        max_consecutive_losses=int(risk.get("max_consecutive_losses", 3)),
        max_daily_losses=int(risk.get("max_daily_losses", 5)),
        max_daily_trades=int(risk.get("max_daily_trades", 0)),
        cooldown_minutes=int(risk.get("cooldown_after_losses_minutes", 30)),
        risk_per_trade=float(risk.get("risk_per_trade", 0.005)),
        max_daily_drawdown=float(risk.get("max_daily_drawdown", 0.03)),
        max_total_drawdown=float(risk.get("max_total_drawdown", 0.08)),
    )


class LossCircuitBreaker:
    """Tracks trade outcomes and pauses new entries when limits are breached."""

    def __init__(self, limits: TradingLimits | None = None, *, now: Callable | None = None):  # type: ignore[valid-type]
        self.limits = limits or TradingLimits()
        self._now = now or (lambda: datetime.now(UTC))
        self.consecutive_losses = 0
        self.daily_losses = 0
        self.daily_trades = 0
        self._day: date | None = None
        self._paused_until: datetime | None = None

    # -- lifecycle ----------------------------------------------------------
    def _roll_day(self, today: date) -> None:
        if self._day != today:
            self._day = today
            self.daily_losses = 0
            self.daily_trades = 0
            # A fresh session resets the consecutive-loss streak but not a
            # safety pause already in force (that has to expire on its own).
            self.consecutive_losses = 0

    def check(self, *, now: datetime | None = None) -> BreakerDecision:
        """Return whether a *new entry* is allowed right now."""
        current = now or self._now()
        self._roll_day(current.date())
        if self._paused_until is not None:
            if current < self._paused_until:
                return BreakerDecision(False, "cooldown after consecutive losses")
            self._paused_until = None
        if self.limits.max_daily_trades > 0 and self.daily_trades >= self.limits.max_daily_trades:
            return BreakerDecision(False, "max daily trades reached")
        if self.limits.max_daily_losses > 0 and self.daily_losses >= self.limits.max_daily_losses:
            return BreakerDecision(False, "max daily losses reached")
        return BreakerDecision(True)

    # -- outcomes -----------------------------------------------------------
    def record_trade(self, realized_pnl: float, *, now: datetime | None = None) -> BreakerDecision:
        """Record a closed trade outcome and return the post-trade state."""
        current = now or self._now()
        self._roll_day(current.date())
        self.daily_trades += 1
        if realized_pnl < 0:
            self.daily_losses += 1
            self.consecutive_losses += 1
            if (
                self.limits.max_consecutive_losses > 0
                and self.consecutive_losses >= self.limits.max_consecutive_losses
            ):
                self._paused_until = current + timedelta(minutes=self.limits.cooldown_minutes)
                self.consecutive_losses = 0
                return BreakerDecision(False, "cooldown after consecutive losses")
        elif realized_pnl > 0:
            self.consecutive_losses = 0
        return self.check(now=current)

    # -- introspection --------------------------------------------------------
    @property
    def paused(self) -> bool:
        if self._paused_until is None:
            return False
        return self._now() < self._paused_until

    def snapshot(self) -> dict:
        return {
            "consecutive_losses": self.consecutive_losses,
            "daily_losses": self.daily_losses,
            "daily_trades": self.daily_trades,
            "paused": self.paused,
            "paused_until": self._paused_until.isoformat() if self._paused_until else None,
        }
