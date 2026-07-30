from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from slytrade.execution.journal import JsonlJournal
from slytrade.execution.models import ExecutionReport, OrderIntent, OrderStatus


@dataclass
class OrderState:
    intent: OrderIntent
    status: OrderStatus = OrderStatus.CREATED
    broker_order_id: str | None = None
    filled_volume: float = 0.0
    avg_fill_price: float | None = None
    message: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def client_order_id(self) -> str:
        return self.intent.client_order_id

    @property
    def is_open(self) -> bool:
        return self.status in {OrderStatus.CREATED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED, OrderStatus.UNKNOWN}


class OrderManagementSystem:
    """In-memory OMS with append-only audit journaling.

    The strategy/RL layer creates OrderIntent objects. The OMS owns order state
    and applies broker/simulator ExecutionReport objects. This prevents strategy
    code from pretending an order filled before the execution layer confirms it.
    """

    def __init__(self, journal: JsonlJournal | None = None):
        self.orders: dict[str, OrderState] = {}
        self.journal = journal

    def create_order(self, intent: OrderIntent) -> OrderState:
        existing = self.orders.get(intent.client_order_id)
        if existing is not None:
            return existing
        state = OrderState(intent=intent)
        self.orders[intent.client_order_id] = state
        self._append("order_created", {"order": state})
        return state

    def apply_report(self, report: ExecutionReport) -> OrderState:
        state = self.orders.get(report.client_order_id)
        if state is None:
            raise KeyError(f"unknown client_order_id: {report.client_order_id}")

        state.status = report.status
        state.broker_order_id = report.broker_order_id or state.broker_order_id
        state.filled_volume = report.filled_volume
        state.avg_fill_price = report.avg_fill_price
        state.message = report.message
        state.updated_at = report.event_time
        self._append("execution_report", {"order": state, "report": report})
        return state

    def get(self, client_order_id: str) -> OrderState | None:
        return self.orders.get(client_order_id)

    def open_orders(self) -> list[OrderState]:
        return [state for state in self.orders.values() if state.is_open]

    def closed_orders(self) -> list[OrderState]:
        return [state for state in self.orders.values() if not state.is_open]

    def _append(self, event_type: str, payload: dict[str, object]) -> None:
        if self.journal is not None:
            self.journal.append(event_type, payload)
