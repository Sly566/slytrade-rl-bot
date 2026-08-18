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


def _bars_frame(minutes: int = 4 * 24 * 60):
    import pandas as pd

    times = pd.date_range("2026-08-01T00:00:00", periods=minutes // 15, freq="15min", tz="UTC")
    close = 4000.0 + pd.Series(range(len(times)), dtype=float) * 0.05
    return pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",
            "timeframe": "M15",
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.4,
            "close": close,
            "tick_volume": 100.0,
            "spread": 0.135,
        }
    )


def test_demo_loop_builds_champion_strategy_with_h4_gate(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_live = True
    settings.stage = TradingStage.DEMO
    settings.timeframe = "M15"
    loop = DemoTradingLoop(settings, FakeMT5())
    strategy = loop.strategy
    # The H4-trend gate is the validated edge; the live loop must not ship the
    # dataclass default (None) that silently skips it.
    assert strategy.config.htf_trend_timeframe == "h4"
    assert strategy.config.min_score == 3  # M15 profile


def test_demo_decision_series_carries_mtf_context(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.allow_live = True
    settings.stage = TradingStage.DEMO
    settings.timeframe = "M15"
    settings.history_bars = 5000
    loop = DemoTradingLoop(settings, FakeMT5())
    loop._window_bars = [row.to_dict() for _, row in _bars_frame().iterrows()]
    decision = loop._decision_series(loop._window_bars[-1])
    assert "mtf_bias" in decision.index
    assert any(c.startswith("htf_") for c in decision.index)
