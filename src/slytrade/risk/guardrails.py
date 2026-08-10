from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from slytrade.execution.models import OrderIntent


@dataclass(frozen=True)
class GuardrailConfig:
    allow_live_trading: bool = False
    max_daily_drawdown: float = 0.03
    max_total_drawdown: float = 0.08
    max_position_volume: float = 1.0
    max_spread_points: float = 50.0


@dataclass(frozen=True)
class GuardrailDecision:
    approved: bool
    reason: str = "OK"


class TradingGuardrails:
    def __init__(self, config: GuardrailConfig, initial_equity: float):
        self.config = config
        self.initial_equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.kill_switch = False
        self.session_date: date | None = None
        self.session_start_equity = float(initial_equity)

    def observe_equity(self, equity: float, *, current_date: date | None = None) -> GuardrailDecision:
        """Track equity and activate the kill switch on configured drawdowns."""
        if equity <= 0:
            self.kill_switch = True
            return GuardrailDecision(False, "equity must remain positive")
        day = current_date or date.today()
        if self.session_date is None:
            self.session_date = day
            self.session_start_equity = self.initial_equity
        elif self.session_date != day:
            self.session_date = day
            self.session_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        daily_dd = (self.session_start_equity - equity) / max(self.session_start_equity, 1e-9)
        total_dd = (self.peak_equity - equity) / max(self.peak_equity, 1e-9)
        if daily_dd >= self.config.max_daily_drawdown:
            self.kill_switch = True
            return GuardrailDecision(False, "max daily drawdown breached")
        if total_dd >= self.config.max_total_drawdown:
            self.kill_switch = True
            return GuardrailDecision(False, "max total drawdown breached")
        return GuardrailDecision(True)

    def approve_order(
        self,
        intent: OrderIntent,
        *,
        equity: float,
        spread_points: float | None = None,
        live: bool = False,
        current_date: date | None = None,
    ) -> GuardrailDecision:
        equity_status = self.observe_equity(equity, current_date=current_date)
        if not equity_status.approved:
            return equity_status
        if self.kill_switch:
            return GuardrailDecision(False, "kill switch active")

        if live and not self.config.allow_live_trading:
            return GuardrailDecision(False, "live trading disabled")

        if intent.volume <= 0:
            return GuardrailDecision(False, "volume must be positive")

        if intent.volume > self.config.max_position_volume:
            return GuardrailDecision(False, "position volume too large")

        if spread_points is not None and spread_points > self.config.max_spread_points:
            return GuardrailDecision(False, "spread too high")

        return GuardrailDecision(True)
