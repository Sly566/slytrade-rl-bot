from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from slytrade.backtest.execution import ExecutionConfig, Quote, TickExecutionSimulator
from slytrade.backtest.metrics import PerformanceMetrics, compute_performance_metrics
from slytrade.backtest.portfolio import Fill, PortfolioState
from slytrade.execution.models import OrderIntent, OrderStatus


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


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: list[float]
    reports: list[object]
    metrics: PerformanceMetrics
    final_portfolio: PortfolioState


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
    """Minimal bar-driven backtest engine with quote-based simulated fills."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.execution = TickExecutionSimulator(
            ExecutionConfig(
                point_size=self.config.point_size,
                point_value=self.config.point_value,
                slippage_points=self.config.slippage_points,
                commission_per_volume=self.config.commission_per_volume,
            )
        )

    def run(self, bars: pd.DataFrame, strategy: BarStrategy) -> BacktestResult:
        required = {"time", "symbol", "open", "high", "low", "close"}
        missing = required.difference(bars.columns)
        if missing:
            raise ValueError(f"bars missing required columns: {sorted(missing)}")

        ordered = bars.sort_values("time").reset_index(drop=True)
        portfolio = PortfolioState(initial_balance=self.config.initial_balance)
        equity_curve = [self.config.initial_balance]
        reports: list[object] = []
        fills = 0
        marks: dict[str, float] = {}

        for index, bar in ordered.iterrows():
            quote = quote_from_bar(
                bar,
                default_spread_points=self.config.default_spread_points,
                point_size=self.config.point_size,
            )
            marks[quote.symbol] = quote.mid
            intent = strategy.on_bar(index, bar)
            if intent is not None:
                simulated = self.execution.execute(intent, quote)
                reports.append(simulated.report)
                if simulated.report.status == OrderStatus.FILLED and simulated.report.avg_fill_price is not None:
                    portfolio.apply_fill(
                        Fill(
                            symbol=intent.symbol,
                            side=intent.side,
                            volume=simulated.report.filled_volume,
                            price=simulated.report.avg_fill_price,
                            commission=simulated.commission,
                            point_value=simulated.point_value,
                        )
                    )
                    fills += 1
            equity_curve.append(portfolio.mark_to_market(marks))

        metrics = compute_performance_metrics(equity_curve, trades=fills)
        return BacktestResult(equity_curve=equity_curve, reports=reports, metrics=metrics, final_portfolio=portfolio)
