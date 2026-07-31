from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from slytrade.backtest.execution import ExecutionConfig, Quote
from slytrade.backtest.metrics import PerformanceMetrics, compute_performance_metrics
from slytrade.backtest.portfolio import PortfolioState
from slytrade.execution.ledger import TradeRecord
from slytrade.execution.models import ExecutionReport, OrderIntent
from slytrade.execution.oms import OrderState
from slytrade.execution.paper_broker import PaperBroker
from slytrade.risk.guardrails import GuardrailConfig


class BarStrategy(Protocol):
    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        """Return an order intent for the current bar, or None."""


@dataclass(frozen=True)
class BacktestConfig:
    initial_balance: float = 100_000.0
    default_spread_points: float = 20.0
    point_size: float = 0.01
    point_value: float = 1.0
    slippage_points: float = 0.0
    commission_per_volume: float = 0.0
    max_spread_points: float = 1_000.0
    max_position_volume: float = 100.0
    max_quote_age_seconds: float = 5.0
    allow_bar_quote_fallback: bool = True


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: list[float]
    reports: list[ExecutionReport]
    metrics: PerformanceMetrics
    final_portfolio: PortfolioState
    orders: list[OrderState]
    trades: list[TradeRecord]


@dataclass
class BuyAndHoldOnceStrategy:
    symbol: str
    volume: float
    _submitted: bool = field(default=False, init=False)

    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        if self._submitted:
            return None
        self._submitted = True
        from slytrade.execution.models import Side

        return OrderIntent(symbol=self.symbol, side=Side.BUY, volume=self.volume, reason="buy_and_hold_once")


def quote_from_bar(bar: pd.Series, *, default_spread_points: float, point_size: float) -> Quote:
    close = float(bar["close"])
    spread_value = float(bar.get("spread", 0.0))
    # MT5 bars often report spread in points. If spread is absent/zero, use configured default.
    spread_price = (spread_value if spread_value > 0 else default_spread_points) * point_size
    bid = close - spread_price / 2.0
    ask = close + spread_price / 2.0
    return Quote(symbol=str(bar["symbol"]), bid=bid, ask=ask, time=pd.Timestamp(bar["time"]).to_pydatetime())


class BarBacktestEngine:
    """Bar-driven backtest engine routed through the production paper path.

    Orders flow through:

    Strategy -> OrderIntent -> PaperBroker -> Guardrails -> OMS -> Execution -> Portfolio -> Ledger
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def make_broker(self) -> PaperBroker:
        return PaperBroker(
            initial_balance=self.config.initial_balance,
            execution_config=ExecutionConfig(
                point_size=self.config.point_size,
                point_value=self.config.point_value,
                slippage_points=self.config.slippage_points,
                commission_per_volume=self.config.commission_per_volume,
            ),
            guardrail_config=GuardrailConfig(
                max_spread_points=self.config.max_spread_points,
                max_position_volume=self.config.max_position_volume,
            ),
        )

    def run(self, bars: pd.DataFrame, strategy: BarStrategy) -> BacktestResult:
        required = {"time", "symbol", "open", "high", "low", "close"}
        missing = required.difference(bars.columns)
        if missing:
            raise ValueError(f"bars missing required columns: {sorted(missing)}")

        ordered = bars.sort_values("time").reset_index(drop=True)
        broker = self.make_broker()
        equity_curve = [self.config.initial_balance]
        reports: list[ExecutionReport] = []

        for index, bar in ordered.iterrows():
            quote = quote_from_bar(
                bar,
                default_spread_points=self.config.default_spread_points,
                point_size=self.config.point_size,
            )
            broker.update_quote(quote)
            intent = strategy.on_bar(index, bar)
            if intent is not None:
                result = broker.submit_order(intent, quote)
                reports.append(result.report)
            equity_curve.append(broker.portfolio.mark_to_market(broker.last_marks))

        metrics = compute_performance_metrics(equity_curve, trades=len(broker.ledger.records))
        return BacktestResult(
            equity_curve=equity_curve,
            reports=reports,
            metrics=metrics,
            final_portfolio=broker.portfolio,
            orders=list(broker.oms.orders.values()),
            trades=list(broker.ledger.records),
        )
