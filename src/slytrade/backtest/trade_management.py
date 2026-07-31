from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from slytrade.backtest.aligned_engine import quote_from_aligned_bar
from slytrade.backtest.engine import BacktestConfig, BacktestResult, BarStrategy
from slytrade.backtest.execution import Quote
from slytrade.backtest.metrics import compute_performance_metrics
from slytrade.execution.models import ExecutionReport, OrderIntent, Side
from slytrade.execution.paper_broker import PaperBroker

ExitReason = Literal[
    "stop_loss",
    "partial_take_profit",
    "take_profit",
    "trailing_stop",
    "max_bars",
    "none",
]


@dataclass(frozen=True)
class TradeManagementConfig:
    """Deterministic managed-trade configuration.

    The engine is deliberately conservative. If SL and TP are touched in the
    same bar, SL wins by default because the intra-bar sequence is unknown in
    aligned-bar mode.
    """

    stop_loss_atr: float = 1.0
    take_profit_atr: float = 2.0
    min_stop_distance: float = 0.10
    max_bars_in_trade: int | None = None
    conservative_same_bar_exit: bool = True
    partial_take_profit_enabled: bool = False
    partial_take_profit_atr: float = 1.0
    partial_close_fraction: float = 0.5
    move_to_breakeven_after_partial: bool = True
    trailing_stop_atr: float | None = None

    def __post_init__(self) -> None:
        if self.stop_loss_atr <= 0:
            raise ValueError("stop_loss_atr must be positive")
        if self.take_profit_atr <= 0:
            raise ValueError("take_profit_atr must be positive")
        if self.min_stop_distance <= 0:
            raise ValueError("min_stop_distance must be positive")
        if self.partial_take_profit_atr <= 0:
            raise ValueError("partial_take_profit_atr must be positive")
        if not 0 < self.partial_close_fraction < 1:
            raise ValueError("partial_close_fraction must be between 0 and 1")
        if self.trailing_stop_atr is not None and self.trailing_stop_atr <= 0:
            raise ValueError("trailing_stop_atr must be positive when provided")


@dataclass
class ManagedTradeState:
    symbol: str
    side: Side
    initial_volume: float
    remaining_volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    partial_take_profit: float | None
    entry_index: int
    partial_taken: bool = False
    breakeven_applied: bool = False

    @property
    def is_long(self) -> bool:
        return self.side == Side.BUY

    @property
    def exit_side(self) -> Side:
        return Side.SELL if self.is_long else Side.BUY

    @property
    def is_closed(self) -> bool:
        return self.remaining_volume <= 1e-12


def risk_unit_from_bar(bar: pd.Series, config: TradeManagementConfig) -> float:
    atr = float(bar.get("atr", 0.0) or 0.0)
    return max(atr * config.stop_loss_atr, config.min_stop_distance)


def target_unit_from_bar(bar: pd.Series, config: TradeManagementConfig) -> float:
    atr = float(bar.get("atr", 0.0) or 0.0)
    return max(atr * config.take_profit_atr, config.min_stop_distance)


def partial_target_unit_from_bar(bar: pd.Series, config: TradeManagementConfig) -> float:
    atr = float(bar.get("atr", 0.0) or 0.0)
    return max(atr * config.partial_take_profit_atr, config.min_stop_distance)


def create_trade_state(intent: OrderIntent, entry_price: float, bar: pd.Series, index: int, config: TradeManagementConfig) -> ManagedTradeState:
    stop_distance = risk_unit_from_bar(bar, config)
    target_distance = target_unit_from_bar(bar, config)
    partial_distance = partial_target_unit_from_bar(bar, config)
    if intent.side == Side.BUY:
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + target_distance
        partial_take_profit = entry_price + partial_distance if config.partial_take_profit_enabled else None
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - target_distance
        partial_take_profit = entry_price - partial_distance if config.partial_take_profit_enabled else None
    return ManagedTradeState(
        symbol=intent.symbol,
        side=intent.side,
        initial_volume=intent.volume,
        remaining_volume=intent.volume,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        partial_take_profit=partial_take_profit,
        entry_index=index,
    )


def _bar_high_low(bar: pd.Series) -> tuple[float, float]:
    high = float(bar.get("tick_mid_high", bar.get("high", 0.0)) or 0.0)
    low = float(bar.get("tick_mid_low", bar.get("low", 0.0)) or 0.0)
    return high, low


def _level_hit(trade: ManagedTradeState, level: float, high: float, low: float) -> bool:
    return high >= level if trade.is_long else low <= level


def _stop_hit(trade: ManagedTradeState, high: float, low: float) -> bool:
    return low <= trade.stop_loss if trade.is_long else high >= trade.stop_loss


def update_trailing_stop(trade: ManagedTradeState, bar: pd.Series, config: TradeManagementConfig) -> None:
    if config.trailing_stop_atr is None:
        return
    atr = float(bar.get("atr", 0.0) or 0.0)
    distance = max(atr * config.trailing_stop_atr, config.min_stop_distance)
    high, low = _bar_high_low(bar)
    if trade.is_long:
        trade.stop_loss = max(trade.stop_loss, high - distance)
    else:
        trade.stop_loss = min(trade.stop_loss, low + distance)


def exit_reason_for_bar(trade: ManagedTradeState, bar: pd.Series, index: int, config: TradeManagementConfig) -> ExitReason:
    event = next_exit_event(trade, bar, index, config)
    return event[0]


def next_exit_event(
    trade: ManagedTradeState,
    bar: pd.Series,
    index: int,
    config: TradeManagementConfig,
) -> tuple[ExitReason, float, float | None]:
    """Return the next exit event as (reason, volume, price_or_none)."""
    high, low = _bar_high_low(bar)
    stop_hit = _stop_hit(trade, high, low)
    final_target_hit = _level_hit(trade, trade.take_profit, high, low)
    partial_hit = (
        trade.partial_take_profit is not None
        and not trade.partial_taken
        and _level_hit(trade, trade.partial_take_profit, high, low)
    )

    if stop_hit and (final_target_hit or partial_hit) and config.conservative_same_bar_exit:
        return "stop_loss", trade.remaining_volume, trade.stop_loss
    if stop_hit:
        return "stop_loss", trade.remaining_volume, trade.stop_loss
    if partial_hit and trade.partial_take_profit is not None:
        partial_volume = min(trade.remaining_volume, trade.initial_volume * config.partial_close_fraction)
        return "partial_take_profit", partial_volume, trade.partial_take_profit
    if final_target_hit:
        return "take_profit", trade.remaining_volume, trade.take_profit
    if config.max_bars_in_trade is not None and index - trade.entry_index >= config.max_bars_in_trade:
        return "max_bars", trade.remaining_volume, None
    return "none", 0.0, None


def quote_for_exit_price(trade: ManagedTradeState, exit_price: float, reference_quote: Quote) -> Quote:
    """Build a quote that fills the exit at a deterministic level.

    The simulator fills sells at bid and buys at ask, so set the relevant side
    to the desired exit price and preserve the current spread on the other side.
    """
    spread = max(reference_quote.spread, 0.0)
    if trade.exit_side == Side.SELL:
        return Quote(symbol=trade.symbol, bid=exit_price, ask=exit_price + spread, time=reference_quote.time)
    return Quote(symbol=trade.symbol, bid=exit_price - spread, ask=exit_price, time=reference_quote.time)


class ManagedAlignedBacktestEngine:
    """Aligned backtest engine with SL/TP, partials, breakeven and trailing."""

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
                update_trailing_stop(trade_state, bar, self.trade_config)
                # A bar can trigger a partial and later another bar can trigger a final exit.
                reason, volume, exit_price = next_exit_event(trade_state, bar, index, self.trade_config)
                if reason != "none" and volume > 0:
                    exit_quote = quote_for_exit_price(trade_state, exit_price, quote) if exit_price is not None else quote
                    exit_intent = OrderIntent(
                        symbol=trade_state.symbol,
                        side=trade_state.exit_side,
                        volume=volume,
                        reason=f"managed_{reason}",
                    )
                    exit_result = broker.submit_order(exit_intent, exit_quote)
                    reports.append(exit_result.report)
                    if exit_result.report.filled_volume > 0:
                        trade_state.remaining_volume -= exit_result.report.filled_volume
                        if reason == "partial_take_profit":
                            trade_state.partial_taken = True
                            if self.trade_config.move_to_breakeven_after_partial:
                                trade_state.stop_loss = trade_state.entry_price
                                trade_state.breakeven_applied = True
                        if trade_state.is_closed:
                            trade_state = None

            if trade_state is None and not broker.guardrails.kill_switch:
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
