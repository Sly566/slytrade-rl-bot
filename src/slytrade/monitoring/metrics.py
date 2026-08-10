from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ExecutionMetrics:
    """Thread-safe counters used by adapters and demo-run reporting."""

    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    broker_errors: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def submitted(self) -> None:
        with self._lock:
            self.orders_submitted += 1

    def filled(self) -> None:
        with self._lock:
            self.orders_filled += 1

    def rejected(self) -> None:
        with self._lock:
            self.orders_rejected += 1

    def broker_error(self) -> None:
        with self._lock:
            self.broker_errors += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "orders_submitted": self.orders_submitted,
                "orders_filled": self.orders_filled,
                "orders_rejected": self.orders_rejected,
                "broker_errors": self.broker_errors,
            }
