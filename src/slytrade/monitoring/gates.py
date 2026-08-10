from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DeploymentStage(StrEnum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    SHADOW = "shadow"
    DEMO = "demo"


@dataclass(frozen=True)
class DeploymentGate:
    stage: DeploymentStage
    required_checks: tuple[str, ...]
    completed_checks: frozenset[str] = frozenset()

    @property
    def approved(self) -> bool:
        return bool(self.required_checks) and set(self.required_checks) <= set(self.completed_checks)

    def complete(self, check: str) -> DeploymentGate:
        if check not in self.required_checks:
            raise ValueError(f"unknown deployment check: {check}")
        return DeploymentGate(self.stage, self.required_checks, self.completed_checks | {check})


DEFAULT_DEMO_GATE = DeploymentGate(
    stage=DeploymentStage.DEMO,
    required_checks=(
        "python312",
        "tests",
        "lint",
        "type_check",
        "historical_validation",
        "cost_stress",
        "seed_stability",
        "lockbox_test",
        "paper_stability",
        "mt5_reconciliation",
        "rollback_verified",
        "manual_approval",
    ),
)
