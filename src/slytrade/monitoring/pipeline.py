"""Fail-closed end-to-end pipeline readiness benchmarking."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from slytrade.data.validators import ValidationReport
from slytrade.monitoring.gates import DeploymentGate
from slytrade.monitoring.health import HealthRegistry


class PipelineStage(StrEnum):
    INGESTION = "ingestion"
    VALIDATION = "validation"
    FEATURES = "features"
    EVALUATION = "evaluation"
    GOVERNANCE = "governance"
    EXECUTION = "execution"
    OPERATIONS = "operations"


@dataclass(frozen=True)
class PipelineCheck:
    stage: PipelineStage
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PipelineBenchmark:
    checks: tuple[PipelineCheck, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def failed(self) -> tuple[PipelineCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def benchmark_pipeline(
    *,
    bars: pd.DataFrame,
    features: pd.DataFrame,
    validation: ValidationReport,
    gate: DeploymentGate,
    health: HealthRegistry,
    model_registry_valid: bool,
    required_checks: set[str] | None = None,
) -> PipelineBenchmark:
    """Benchmark every hand-off without executing or placing an order."""
    checks = [
        PipelineCheck(PipelineStage.INGESTION, "bars_present", not bars.empty, f"{len(bars)} bars"),
        PipelineCheck(
            PipelineStage.VALIDATION,
            "validated_data",
            validation.valid and validation.rows_after > 0,
            f"{validation.rows_after}/{validation.rows_before} rows accepted",
        ),
        PipelineCheck(
            PipelineStage.FEATURES,
            "features_aligned",
            len(features) == len(bars) and not features.empty,
            f"{len(features)} feature rows for {len(bars)} bars",
        ),
        PipelineCheck(PipelineStage.GOVERNANCE, "model_registry", model_registry_valid, "hash chain verification"),
        PipelineCheck(PipelineStage.EXECUTION, "deployment_gate", gate.approved, f"stage={gate.stage}"),
        PipelineCheck(PipelineStage.OPERATIONS, "health_registry", health.is_ready(), "all health checks healthy"),
    ]
    if required_checks is not None:
        missing = required_checks.difference(gate.completed_checks)
        checks.append(
            PipelineCheck(
                PipelineStage.EVALUATION,
                "required_evidence",
                not missing,
                "all required evidence present" if not missing else f"missing: {sorted(missing)}",
            )
        )
    return PipelineBenchmark(tuple(checks))
