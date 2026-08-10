from datetime import UTC, datetime, timedelta

from slytrade.monitoring.gates import DeploymentStage
from slytrade.monitoring.operations import PersistentKillSwitch, RollbackArtifact, SoakMonitor
from slytrade.risk.guardrails import GuardrailConfig, TradingGuardrails


def test_kill_switch_survives_guardrail_restart(tmp_path):
    artifact = tmp_path / "kill-switch.json"
    guardrails = TradingGuardrails(
        GuardrailConfig(max_daily_drawdown=0.03),
        initial_equity=100_000,
        kill_switch_path=artifact,
    )
    guardrails.observe_equity(96_000, current_date=datetime(2026, 1, 1, tzinfo=UTC).date())

    restored = TradingGuardrails(GuardrailConfig(), initial_equity=100_000, kill_switch_path=artifact)
    assert restored.kill_switch
    assert PersistentKillSwitch(artifact).reason == "max daily drawdown breached"
    restored.clear_kill_switch()
    assert not TradingGuardrails(GuardrailConfig(), initial_equity=100_000, kill_switch_path=artifact).kill_switch


def test_rollback_artifact_round_trip(tmp_path):
    artifact = RollbackArtifact("demo-2026-08-10", metadata={"config": "stable"})
    path = tmp_path / "rollback.json"
    artifact.save(path)
    restored = RollbackArtifact.load(path)
    assert restored.version == artifact.version
    assert restored.metadata == artifact.metadata


def test_shadow_soak_alerts_on_errors_and_stale_heartbeat():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    monitor = SoakMonitor(DeploymentStage.SHADOW, min_samples=2, max_error_rate=0.5, stale_after=timedelta(minutes=5))
    monitor.record(healthy=True, now=now)
    assert not monitor.ready
    alerts = monitor.record(healthy=False, detail="quote gap", now=now + timedelta(minutes=1))
    assert {alert.code for alert in alerts} == {"soak_unhealthy"}
    assert not monitor.ready
    stale = monitor.check_stale(now=now + timedelta(minutes=7))
    assert any(alert.code == "soak_stale" for alert in stale)
