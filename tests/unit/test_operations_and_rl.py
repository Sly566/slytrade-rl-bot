import pandas as pd
import pytest

from slytrade.monitoring.gates import DEFAULT_DEMO_GATE
from slytrade.monitoring.health import HealthRegistry
from slytrade.monitoring.metrics import ExecutionMetrics
from slytrade.rl.environment import TradingEnvironment


def test_demo_gate_requires_all_checks():
    assert not DEFAULT_DEMO_GATE.approved
    completed = DEFAULT_DEMO_GATE
    for check in DEFAULT_DEMO_GATE.required_checks:
        completed = completed.complete(check)
    assert completed.approved


def test_health_registry_and_metrics():
    health = HealthRegistry()
    health.report("market_data", True, "connected")
    health.report("broker", False, "not reconciled")
    assert not health.is_ready()
    metrics = ExecutionMetrics()
    metrics.submitted()
    metrics.rejected()
    assert metrics.snapshot()["orders_submitted"] == 1
    assert metrics.snapshot()["orders_rejected"] == 1


def test_rl_environment_requires_optional_dependency_or_runs():
    bars = pd.DataFrame({"close": [100.0, 101.0, 100.5]})
    try:
        env = TradingEnvironment(bars)
    except ImportError:
        pytest.skip("RL dependencies are not installed")
    observation, _ = env.reset(seed=7)
    assert observation.shape == (4,)
    _, reward, _, _, info = env.step(2)
    assert isinstance(reward, float)
    assert info["equity"] > 0
