from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class HealthStatus:
    name: str
    healthy: bool
    detail: str
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class HealthRegistry:
    """Small dependency-free health registry for readiness and liveness checks."""

    statuses: dict[str, HealthStatus] = field(default_factory=dict)

    def report(self, name: str, healthy: bool, detail: str) -> HealthStatus:
        status = HealthStatus(name=name, healthy=healthy, detail=detail)
        self.statuses[name] = status
        return status

    def is_ready(self) -> bool:
        return bool(self.statuses) and all(status.healthy for status in self.statuses.values())

    def stale(self, name: str, max_age: timedelta, *, now: datetime | None = None) -> bool:
        status = self.statuses.get(name)
        if status is None:
            return True
        current = now or datetime.now(UTC)
        return current - status.checked_at > max_age
