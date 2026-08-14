from __future__ import annotations

from pathlib import Path

from slytrade import tasks
from slytrade.rl.environment import RLEnvironmentConfig


def test_rl_env_config_has_reward_fields() -> None:
    config = RLEnvironmentConfig()
    assert config.reward_type == "raw"
    assert config.drawdown_tolerance == 0.05
    adjusted = tasks._with_reward(config, "risk_adjusted")
    assert adjusted.reward_type == "risk_adjusted"


def test_generate_sample_dataset(tmp_path: Path) -> None:
    files = tasks.generate_sample_dataset("XAUUSD", start="2025-01-01", bar_periods=200, tick_periods=1_000, out_dir=tmp_path)
    assert Path(files["bars"]).exists()
    assert Path(files["ticks"]).exists()


def test_collect_all_samples(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tasks, "SAMPLE_ROOT", str(tmp_path))
    result = tasks.collect_all("XAUUSD", source="samples", sample_start="2025-01-01")
    assert result.ok
    assert result.data["source"] == "samples"


def test_align_and_backtest_on_samples(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tasks, "SAMPLE_ROOT", str(tmp_path))
    tasks.generate_sample_dataset("XAUUSD", start="2025-01-01", bar_periods=400, tick_periods=2_000, out_dir=tmp_path)
    aligned = tasks.align("XAUUSD", timeframe="M1")
    assert aligned.ok, aligned.message
    bars_file = aligned.data["bars_file"]
    result = tasks.backtest(bars_file, strategy="persona-adaptive", symbol="XAUUSD")
    assert result.ok
    assert "total_return" in (result.data or {})


def test_default_point_value() -> None:
    assert tasks.default_point_value("XAUUSD") == 100.0
    assert tasks.default_point_value("XAGUSD") == 100.0
    assert tasks.default_point_value("EURUSD") == 1.0
