from __future__ import annotations

from slytrade.runtime.settings import RuntimeSettings, TradingStage


def test_defaults_are_fail_closed() -> None:
    settings = RuntimeSettings()
    assert settings.allow_live is False
    assert settings.stage == TradingStage.PAPER
    assert settings.metrics_port == 9108
    assert settings.fail_closed_checks() == []


def test_live_without_demo_stage_blocks_startup() -> None:
    settings = RuntimeSettings(allow_live=True, stage=TradingStage.PAPER)
    problems = settings.fail_closed_checks()
    assert any("ALLOW_LIVE" in problem for problem in problems)


def test_invalid_port_blocks_startup() -> None:
    settings = RuntimeSettings(metrics_port=0)
    assert any("metrics_port" in problem for problem in settings.fail_closed_checks())


def test_trading_days_parsed() -> None:
    settings = RuntimeSettings(trading_days="mon,tue,wed")
    assert settings.trading_days_set == frozenset({"mon", "tue", "wed"})
