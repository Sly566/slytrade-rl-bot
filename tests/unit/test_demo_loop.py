from __future__ import annotations

import pytest

from slytrade.runtime.demo_loop import DemoTradingLoop
from slytrade.runtime.settings import RuntimeSettings, TradingStage


class FakeMT5:
    def __init__(self) -> None:
        self.initialized = True

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def symbols_get(self):
        return []

    def symbol_select(self, name: str, enable: bool = True) -> bool:
        return True


def _settings(tmp_path) -> RuntimeSettings:
    return RuntimeSettings(
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        kill_switch_path=str(tmp_path / "state" / "kill-switch.json"),
        json_logs=False,
        symbol="XAUUSD",
        timeframe="M1",
        poll_seconds=0.01,
    )


def test_demo_loop_refuses_without_allow_live(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_live = False
    with pytest.raises(ValueError, match="live trading is disabled"):
        DemoTradingLoop(settings, FakeMT5())


def test_demo_loop_refuses_non_demo_stage(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_live = True
    settings.stage = TradingStage.PAPER
    with pytest.raises(ValueError, match="requires stage=demo"):
        DemoTradingLoop(settings, FakeMT5())


def test_demo_loop_refuses_startup_problems(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_live = True
    settings.stage = TradingStage.DEMO
    settings.metrics_port = 0  # invalid
    with pytest.raises(ValueError, match="startup blocked"):
        DemoTradingLoop(settings, FakeMT5())


def test_demo_loop_constructs_when_authorized(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_live = True
    settings.stage = TradingStage.DEMO
    loop = DemoTradingLoop(settings, FakeMT5())
    assert loop.adapter.allow_trading is True
    assert loop.guardrails.config.allow_live_trading is True
