from __future__ import annotations

import pandas as pd

from slytrade.backtest.engine import BacktestConfig, BacktestResult, BarStrategy
from slytrade.backtest.execution import ExecutionConfig, Quote
from slytrade.backtest.metrics import compute_performance_metrics
from slytrade.execution.models import ExecutionReport
from slytrade.execution.paper_broker import PaperBroker
from slytrade.risk.guardrails import GuardrailConfig


def quote_from_aligned_bar(bar: pd.Series) -> Quote | None:
    required = {"quote_bid", "quote_ask", "quote_time"}
    if not required.issubset(set(bar.index)):
        return None
    if pd.isna(bar["quote_bid"]) or pd.isna(bar["quote_ask"]) or pd.isna(bar["quote_time"]):
        return None
    return Quote(
        symbol=str(bar["symbol"]),
        bid=float(bar["quote_bid"]),
        ask=float(bar["quote_ask"]),
        time=pd.Timestamp(bar["quote_time"]).to_pydatetime(),
    )


class AlignedBacktestEngine:
    """Backtest engine for aligned datasets with precomputed decision quotes."""

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

    def run(self, aligned_bars: pd.DataFrame, strategy: BarStrategy) -> BacktestResult:
        required = {"time", "symbol", "open", "high", "low", "close", "decision_time"}
        missing = required.difference(aligned_bars.columns)
        if missing:
            raise ValueError(f"aligned bars missing required columns: {sorted(missing)}")

        ordered = aligned_bars.sort_values("decision_time").reset_index(drop=True)
        broker = self.make_broker()
        equity_curve = [self.config.initial_balance]
        reports: list[ExecutionReport] = []

        for index, bar in ordered.iterrows():
            quote = quote_from_aligned_bar(bar)
            quote_fresh = bool(bar.get("quote_is_fresh", False))
            if quote is not None and quote_fresh:
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
