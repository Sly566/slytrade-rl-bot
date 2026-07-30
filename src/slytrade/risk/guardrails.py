from __future__ import annotations

from dataclasses import dataclass

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

    def approve_order(
        self,
        intent: OrderIntent,
        *,
        equity: float,
        spread_points: float | None = None,
        live: bool = False,
    ) -> GuardrailDecision:
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

        self.peak_equity = max(self.peak_equity, equity)
        total_dd = (self.peak_equity - equity) / max(self.peak_equity, 1e-9)

        if total_dd >= self.config.max_total_drawdown:
            self.kill_switch = True
            return GuardrailDecision(False, "max total drawdown breached")

        return GuardrailDecision(True)
