from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from slytrade.backtest.aligned_engine import quote_from_aligned_bar
from slytrade.backtest.engine import BacktestConfig, BacktestResult, BarStrategy
from slytrade.backtest.metrics import compute_performance_metrics
from slytrade.execution.models import ExecutionReport, OrderIntent, Side
from slytrade.execution.paper_broker import PaperBroker

ExitReason = Literal["stop_loss", "take_profit", "max_bars", "none"]


@dataclass(frozen=True)
class TradeManagementConfig:
    """Simple one-position trade management configuration.

    This is intentionally deterministic and conservative. If both stop-loss and
    take-profit are touched within the same bar, stop-loss wins by default.
    """

    stop_loss_atr: float = 1.0
    take_profit_atr: float = 2.0
    min_stop_distance: float = 0.10
    max_bars_in_trade: int | None = None
    conservative_same_bar_exit: bool = True


@dataclass
class ManagedTradeState:
    symbol: str
    side: Side
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_index: int

    @property
    def is_long(self) -> bool:
        return self.side == Side.BUY

    @property
    def exit_side(self) -> Side:
        return Side.SELL if self.is_long else Side.BUY


def risk_unit_from_bar(bar: pd.Series, config: TradeManagementConfig) -> float:
    atr = float(bar.get("atr", 0.0) or 0.0)
    return max(atr * config.stop_loss_atr, config.min_stop_distance)


def target_unit_from_bar(bar: pd.Series, config: TradeManagementConfig) -> float:
    atr = float(bar.get("atr", 0.0) or 0.0)
    return max(atr * config.take_profit_atr, config.min_stop_distance)


def create_trade_state(intent: OrderIntent, entry_price: float, bar: pd.Series, index: int, config: TradeManagementConfig) -> ManagedTradeState:
    stop_distance = risk_unit_from_bar(bar, config)
    target_distance = target_unit_from_bar(bar, config)
    if intent.side == Side.BUY:
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + target_distance
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - target_distance
    return ManagedTradeState(
        symbol=intent.symbol,
        side=intent.side,
        volume=intent.volume,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry_index=index,
    )


def exit_reason_for_bar(trade: ManagedTradeState, bar: pd.Series, index: int, config: TradeManagementConfig) -> ExitReason:
    high = float(bar.get("tick_mid_high", bar.get("high", 0.0)) or 0.0)
    low = float(bar.get("tick_mid_low", bar.get("low", 0.0)) or 0.0)

    if trade.is_long:
        stop_hit = low <= trade.stop_loss
        target_hit = high >= trade.take_profit
    else:
        stop_hit = high >= trade.stop_loss
        target_hit = low <= trade.take_profit

    if stop_hit and target_hit:
        return "stop_loss" if config.conservative_same_bar_exit else "take_profit"
    if stop_hit:
        return "stop_loss"
    if target_hit:
        return "take_profit"
    if config.max_bars_in_trade is not None and index - trade.entry_index >= config.max_bars_in_trade:
        return "max_bars"
    return "none"


class ManagedAlignedBacktestEngine:
    """Aligned backtest engine with basic SL/TP trade management.

    The entry strategy proposes entries only. This engine owns trade exits.
    """

    def __init__(self, config: BacktestConfig | None = None, trade_config: TradeManagementConfig | None = None):
        self.config = config or BacktestConfig()
        self.trade_config = trade_config or TradeManagementConfig()
        self.last_trade_state: ManagedTradeState | None = None

    def make_broker(self) -> PaperBroker:
        from slytrade.backtest.aligned_engine import AlignedBacktestEngine

        return AlignedBacktestEngine(self.config).make_broker()

    def run(self, aligned_bars: pd.DataFrame, strategy: BarStrategy) -> BacktestResult:
        required = {"time", "symbol", "open", "high", "low", "close", "decision_time"}
        missing = required.difference(aligned_bars.columns)
        if missing:
            raise ValueError(f"aligned bars missing required columns: {sorted(missing)}")

        ordered = aligned_bars.sort_values("decision_time").reset_index(drop=True)
        broker = self.make_broker()
        equity_curve = [self.config.initial_balance]
        reports: list[ExecutionReport] = []
        trade_state: ManagedTradeState | None = None

        for index, bar in ordered.iterrows():
            quote = quote_from_aligned_bar(bar)
            quote_fresh = bool(bar.get("quote_is_fresh", False))
            if quote is None or not quote_fresh:
                equity_curve.append(broker.portfolio.mark_to_market(broker.last_marks))
                continue

            broker.update_quote(quote)

            if trade_state is not None:
                reason = exit_reason_for_bar(trade_state, bar, index, self.trade_config)
                if reason != "none":
                    exit_intent = OrderIntent(
                        symbol=trade_state.symbol,
                        side=trade_state.exit_side,
                        volume=trade_state.volume,
                        reason=f"managed_{reason}",
                    )
                    exit_result = broker.submit_order(exit_intent, quote)
                    reports.append(exit_result.report)
                    if exit_result.report.filled_volume > 0:
                        trade_state = None

            if trade_state is None:
                intent = strategy.on_bar(index, bar)
                if intent is not None:
                    entry_result = broker.submit_order(intent, quote)
                    reports.append(entry_result.report)
                    if entry_result.report.avg_fill_price is not None and entry_result.report.filled_volume > 0:
                        trade_state = create_trade_state(
                            intent,
                            entry_result.report.avg_fill_price,
                            bar,
                            index,
                            self.trade_config,
                        )

            equity_curve.append(broker.portfolio.mark_to_market(broker.last_marks))

        self.last_trade_state = trade_state
        metrics = compute_performance_metrics(equity_curve, trades=len(broker.ledger.records))
        return BacktestResult(
            equity_curve=equity_curve,
            reports=reports,
            metrics=metrics,
            final_portfolio=broker.portfolio,
            orders=list(broker.oms.orders.values()),
            trades=list(broker.ledger.records),
        )
