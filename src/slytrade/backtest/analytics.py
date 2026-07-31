from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from slytrade.backtest.engine import BacktestResult
from slytrade.execution.ledger import TradeRecord
from slytrade.execution.models import OrderStatus
from slytrade.execution.oms import OrderState


@dataclass(frozen=True)
class TradeAnalytics:
    fills: int
    entry_fills: int
    exit_fills: int
    net_realized_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    win_rate: float
    average_win: float
    average_loss: float
    expectancy: float
    total_commission: float
    exit_reason_counts: dict[str, int] = field(default_factory=dict)
    order_status_counts: dict[str, int] = field(default_factory=dict)
    order_reject_reasons: dict[str, int] = field(default_factory=dict)


def _status_value(status: OrderStatus | str) -> str:
    if isinstance(status, Enum):
        return str(status.value)
    return str(status)


def _exit_reason(record: TradeRecord) -> str:
    if record.reason.startswith("managed_"):
        return record.reason.removeprefix("managed_")
    return "entry_or_other"


def order_status_counts(orders: list[OrderState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for order in orders:
        key = _status_value(order.status)
        counts[key] = counts.get(key, 0) + 1
    return counts


def order_reject_reasons(orders: list[OrderState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for order in orders:
        if order.status == OrderStatus.REJECTED:
            reason = order.message or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def compute_trade_analytics(records: list[TradeRecord], orders: list[OrderState] | None = None) -> TradeAnalytics:
    orders = orders or []
    exit_records = [record for record in records if record.reason.startswith("managed_")]
    entry_records = [record for record in records if not record.reason.startswith("managed_")]
    wins = [record.realized_pnl for record in exit_records if record.realized_pnl > 0]
    losses = [record.realized_pnl for record in exit_records if record.realized_pnl < 0]
    breakeven = [record.realized_pnl for record in exit_records if record.realized_pnl == 0]

    gross_profit = float(sum(wins))
    gross_loss = float(sum(losses))
    net = float(sum(record.realized_pnl for record in records))
    total_commission = float(sum(record.commission for record in records))
    exit_count = len(exit_records)
    win_rate = len(wins) / exit_count if exit_count else 0.0
    average_win = gross_profit / len(wins) if wins else 0.0
    average_loss = gross_loss / len(losses) if losses else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy = net / exit_count if exit_count else 0.0

    reasons: dict[str, int] = {}
    for record in exit_records:
        reason = _exit_reason(record)
        reasons[reason] = reasons.get(reason, 0) + 1
    if breakeven:
        reasons["breakeven"] = reasons.get("breakeven", 0) + len(breakeven)

    return TradeAnalytics(
        fills=len(records),
        entry_fills=len(entry_records),
        exit_fills=exit_count,
        net_realized_pnl=net,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=float(profit_factor),
        win_rate=float(win_rate),
        average_win=float(average_win),
        average_loss=float(average_loss),
        expectancy=float(expectancy),
        total_commission=total_commission,
        exit_reason_counts=reasons,
        order_status_counts=order_status_counts(orders),
        order_reject_reasons=order_reject_reasons(orders),
    )


def compute_result_analytics(result: BacktestResult) -> TradeAnalytics:
    return compute_trade_analytics(result.trades, result.orders)
