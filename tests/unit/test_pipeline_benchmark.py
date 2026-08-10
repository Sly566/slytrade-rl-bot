import pandas as pd

from slytrade.data.validators import ValidationReport
from slytrade.monitoring.gates import DeploymentGate, DeploymentStage
from slytrade.monitoring.health import HealthRegistry
from slytrade.monitoring.pipeline import benchmark_pipeline


def test_pipeline_benchmark_fails_closed_when_evidence_is_missing():
    bars = pd.DataFrame({"close": [1.0, 2.0]})
    features = pd.DataFrame({"signal": [0.1, 0.2]})
    validation = ValidationReport(rows_before=2, rows_after=2)
    gate = DeploymentGate(DeploymentStage.DEMO, ("tests",), frozenset())
    health = HealthRegistry()
    health.report("mt5", True, "connected")

    report = benchmark_pipeline(
        bars=bars,
        features=features,
        validation=validation,
        gate=gate,
        health=health,
        model_registry_valid=True,
        required_checks={"tests", "lockbox_test"},
    )

    assert not report.passed
    assert any(check.name == "required_evidence" for check in report.failed())
