from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from slytrade.backtest.engine import BacktestConfig, BacktestResult, BarStrategy, quote_from_bar
from slytrade.backtest.execution import ExecutionConfig, Quote
from slytrade.backtest.metrics import compute_performance_metrics
from slytrade.execution.models import ExecutionReport
from slytrade.execution.paper_broker import PaperBroker
from slytrade.risk.guardrails import GuardrailConfig


@dataclass(frozen=True)
class TickBacktestStats:
    ticks_processed: int
    fallback_bar_quotes: int


def quote_from_tick(tick: pd.Series) -> Quote:
    """Build an executable bid/ask Quote from one canonical tick row."""
    time_value = tick["time_msc"] if "time_msc" in tick.index else tick["time"]
    return Quote(
        symbol=str(tick["symbol"]),
        bid=float(tick["bid"]),
        ask=float(tick["ask"]),
        time=pd.Timestamp(time_value).to_pydatetime(),
    )


class TickBacktestEngine:
    """Bar-signal, tick-execution backtest engine.

    Strategy decisions are made on bars, but quotes and fills come from ticks
    where available. The engine processes ticks only up to the current bar's
    timestamp, so it remains causal. This assumes bars are timestamped at the
    time the signal is known (typically bar close in a prepared research set).
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

    def run(self, bars: pd.DataFrame, ticks: pd.DataFrame, strategy: BarStrategy) -> BacktestResult:
        required_bars = {"time", "symbol", "open", "high", "low", "close"}
        missing_bars = required_bars.difference(bars.columns)
        if missing_bars:
            raise ValueError(f"bars missing required columns: {sorted(missing_bars)}")

        required_ticks = {"time_msc", "symbol", "bid", "ask"}
        missing_ticks = required_ticks.difference(ticks.columns)
        if missing_ticks:
            raise ValueError(f"ticks missing required columns: {sorted(missing_ticks)}")

        ordered_bars = bars.sort_values("time").reset_index(drop=True)
        ordered_ticks = ticks.sort_values("time_msc").reset_index(drop=True)
        ordered_ticks["time_msc"] = pd.to_datetime(ordered_ticks["time_msc"], utc=True)
        ordered_bars["time"] = pd.to_datetime(ordered_bars["time"], utc=True)

        broker = self.make_broker()
        equity_curve = [self.config.initial_balance]
        reports: list[ExecutionReport] = []
        tick_index = 0
        last_quote_by_symbol: dict[str, Quote] = {}
        fallback_bar_quotes = 0

        for index, bar in ordered_bars.iterrows():
            bar_time = pd.Timestamp(bar["time"])
            while tick_index < len(ordered_ticks) and pd.Timestamp(ordered_ticks.loc[tick_index, "time_msc"]) <= bar_time:
                tick_quote = quote_from_tick(ordered_ticks.loc[tick_index])
                last_quote_by_symbol[tick_quote.symbol] = tick_quote
                broker.update_quote(tick_quote)
                tick_index += 1

            symbol = str(bar["symbol"])
            quote = last_quote_by_symbol.get(symbol)
            if quote is None:
                quote = quote_from_bar(
                    bar,
                    default_spread_points=self.config.default_spread_points,
                    point_size=self.config.point_size,
                )
                fallback_bar_quotes += 1
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
