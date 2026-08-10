from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from slytrade.monitoring.gates import DeploymentStage


@dataclass(frozen=True)
class OperationalAlert:
    code: str
    severity: str
    detail: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class PersistentKillSwitch:
    """Restart-safe kill switch backed by a small JSON artifact."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()
        self._state = self._read()

    @property
    def active(self) -> bool:
        return bool(self._state.get("active", False))

    @property
    def reason(self) -> str | None:
        return self._state.get("reason")

    def activate(self, reason: str, *, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._state = {
                "active": True,
                "reason": reason,
                "activated_at": datetime.now(UTC).isoformat(),
                "metadata": metadata or {},
            }
            self._write()

    def clear(self) -> None:
        with self._lock:
            self._state = {"active": False}
            self._write()

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"active": False}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(json.dumps(self._state, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)


@dataclass(frozen=True)
class RollbackArtifact:
    """Immutable deployment pointer that can be restored after a failed soak."""

    version: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "created_at": self.created_at.isoformat(),
                    "metadata": self.metadata,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> RollbackArtifact:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class SoakMonitor:
    """Paper/shadow soak primitive with deterministic health alerts."""

    def __init__(
        self,
        stage: DeploymentStage,
        *,
        min_samples: int = 1,
        max_error_rate: float = 0.0,
        stale_after: timedelta = timedelta(minutes=5),
    ):
        if stage not in (DeploymentStage.PAPER, DeploymentStage.SHADOW):
            raise ValueError("soak stage must be paper or shadow")
        if min_samples < 1 or not 0 <= max_error_rate <= 1:
            raise ValueError("invalid soak thresholds")
        self.stage = stage
        self.min_samples = min_samples
        self.max_error_rate = max_error_rate
        self.stale_after = stale_after
        self.samples = 0
        self.errors = 0
        self.last_seen: datetime | None = None
        self.alerts: list[OperationalAlert] = []

    @property
    def error_rate(self) -> float:
        return self.errors / self.samples if self.samples else 0.0

    @property
    def ready(self) -> bool:
        return self.samples >= self.min_samples and not self.alerts

    def record(self, *, healthy: bool, detail: str = "", now: datetime | None = None) -> tuple[OperationalAlert, ...]:
        timestamp = now or datetime.now(UTC)
        self.samples += 1
        self.last_seen = timestamp
        if not healthy:
            self.errors += 1
            self._alert("soak_unhealthy", detail or "unhealthy paper/shadow observation")
        if self.error_rate > self.max_error_rate:
            self._alert("soak_error_rate", f"error rate {self.error_rate:.3f} exceeds {self.max_error_rate:.3f}")
        return tuple(self.alerts)

    def check_stale(self, *, now: datetime | None = None) -> tuple[OperationalAlert, ...]:
        current = now or datetime.now(UTC)
        if self.last_seen is None or current - self.last_seen > self.stale_after:
            self._alert("soak_stale", f"no {self.stage.value} heartbeat within {self.stale_after}")
        return tuple(self.alerts)

    def _alert(self, code: str, detail: str) -> None:
        if not any(alert.code == code for alert in self.alerts):
            self.alerts.append(OperationalAlert(code, "critical" if code == "soak_error_rate" else "warning", detail))
