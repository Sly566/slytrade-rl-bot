"""Dashboard settings must override the bot's runtime knobs (no hardcoded defaults)."""
from __future__ import annotations

from types import SimpleNamespace

from slytrade.runtime.demo_loop import LiveTradingLoop
from slytrade.runtime.settings import RuntimeSettings, TradingStage


class FakeMT5:
    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def symbols_get(self):
        return []

    def symbol_select(self, name: str, enable: bool = True) -> bool:
        return True

    def positions_get(self):
        return []

    def account_info(self):
        return SimpleNamespace(equity=1000.0, balance=1000.0, currency="USD")


def _settings(tmp_path, **overrides) -> RuntimeSettings:
    kwargs = dict(
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        kill_switch_path=str(tmp_path / "state" / "kill-switch.json"),
        json_logs=False,
        symbol="XAUUSD",
        timeframe="M15",
        allow_live=True,
        stage=TradingStage.DEMO,
    )
    kwargs.update(overrides)
    return RuntimeSettings(**kwargs)


def test_risk_override_reaches_breaker(tmp_path) -> None:
    loop = LiveTradingLoop(_settings(tmp_path, risk_per_trade=0.02), FakeMT5())
    assert loop.breaker.limits.risk_per_trade == 0.02


def test_limit_entry_atr_override_reaches_strategy(tmp_path) -> None:
    loop = LiveTradingLoop(_settings(tmp_path, limit_entry_atr=0.5), FakeMT5())
    assert loop.strategy.config.limit_entry_atr == 0.5


def test_max_position_volume_override_reaches_guardrails(tmp_path) -> None:
    loop = LiveTradingLoop(_settings(tmp_path, max_position_volume=3.0), FakeMT5())
    assert loop.guardrails.config.max_position_volume == 3.0


def test_unset_overrides_keep_config_values(tmp_path) -> None:
    # with no overrides, the values come from configs/risk.yaml (0.5% risk, 0.25 ATR)
    loop = LiveTradingLoop(_settings(tmp_path), FakeMT5())
    assert loop.breaker.limits.risk_per_trade == 0.005
    assert loop.strategy.config.limit_entry_atr == 0.25
