from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from slytrade.execution.journal import JsonlJournal, SqliteJournal
from slytrade.execution.models import ExecutionReport, OrderIntent, OrderKind, OrderStatus, Side


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

    def __init__(self, journal: JsonlJournal | SqliteJournal | None = None):
        self.orders: dict[str, OrderState] = {}
        self.journal = journal
        if journal is not None:
            self._restore(journal.read_all())

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

    def _restore(self, events: list[dict[str, object]]) -> None:
        """Rebuild order state from the durable event stream after a restart."""
        for event in events:
            event_type = event.get("event_type")
            if event_type == "order_created":
                raw = event.get("order")
                if not isinstance(raw, dict):
                    continue
                intent = _intent_from_dict(raw.get("intent"))
                if intent is not None:
                    self.orders[intent.client_order_id] = OrderState(intent=intent)
            elif event_type == "execution_report":
                raw_report = event.get("report")
                if isinstance(raw_report, dict):
                    report = _report_from_dict(raw_report)
                    if report is not None and report.client_order_id in self.orders:
                        self._apply_without_journal(report)

    def _apply_without_journal(self, report: ExecutionReport) -> None:
        state = self.orders[report.client_order_id]
        state.status = report.status
        state.broker_order_id = report.broker_order_id or state.broker_order_id
        state.filled_volume = report.filled_volume
        state.avg_fill_price = report.avg_fill_price
        state.message = report.message
        state.updated_at = report.event_time


def _intent_from_dict(value: object) -> OrderIntent | None:
    if not isinstance(value, dict):
        return None
    try:
        return OrderIntent(
            symbol=str(value["symbol"]),
            side=Side(str(value["side"])),
            volume=float(value["volume"]),
            kind=OrderKind(str(value.get("kind", OrderKind.MARKET.value))),
            limit_price=_optional_float(value.get("limit_price")),
            stop_loss=_optional_float(value.get("stop_loss")),
            take_profit=_optional_float(value.get("take_profit")),
            reason=str(value.get("reason", "strategy")),
            client_order_id=str(value["client_order_id"]),
            created_at=datetime.fromisoformat(str(value["created_at"])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _report_from_dict(value: dict[str, object]) -> ExecutionReport | None:
    try:
        return ExecutionReport(
            client_order_id=str(value["client_order_id"]),
            status=OrderStatus(str(value["status"])),
            filled_volume=_optional_float(value.get("filled_volume")) or 0.0,
            avg_fill_price=_optional_float(value.get("avg_fill_price")),
            broker_order_id=str(value["broker_order_id"]) if value.get("broker_order_id") is not None else None,
            message=str(value.get("message", "")),
            event_time=datetime.fromisoformat(str(value["event_time"])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)
