from __future__ import annotations

from pathlib import Path

import pandas as pd

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
    aligned = tasks.align("XAUUSD", timeframe="M1", out_dir=str(tmp_path / "aligned"))
    assert aligned.ok, aligned.message
    bars_file = aligned.data["bars_file"]
    result = tasks.backtest(bars_file, strategy="persona-adaptive", symbol="XAUUSD")
    assert result.ok
    assert "total_return" in (result.data or {})


def test_default_point_value() -> None:
    assert tasks.default_point_value("XAUUSD") == 100.0
    assert tasks.default_point_value("XAGUSD") == 100.0
    assert tasks.default_point_value("EURUSD") == 1.0


def test_symbol_dir_finds_broker_suffix(tmp_path: Path) -> None:
    from slytrade.data.schemas import BAR_COLUMNS

    # MT5 stores bars under the RESOLVED symbol (XAUUSDm).
    base = tmp_path / "mt5_bars" / "symbol=XAUUSDm" / "timeframe=M1" / "year=2026"
    base.mkdir(parents=True)
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
            "symbol": "XAUUSDm",
            "timeframe": "M1",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "tick_volume": 10.0,
            "spread": 20.0,
            "real_volume": 0.0,
        }
    )[BAR_COLUMNS]
    bars.to_parquet(base / "day=01.parquet", index=False)

    # Base-symbol lookup must find the suffix variant.
    assert len(tasks.find_collected_bars("XAUUSD", "M1", root=tmp_path)) == 1
    loaded = tasks.load_collected_bars("XAUUSD", "M1", root=tmp_path)
    assert len(loaded) == 5


def test_symbol_dir_prefers_shortest_suffix(tmp_path: Path) -> None:
    from slytrade.data.schemas import BAR_COLUMNS

    for resolved in ("XAUUSDm", "XAUUSD247m"):
        base = tmp_path / "mt5_bars" / f"symbol={resolved}" / "timeframe=M1" / "y=2026"
        base.mkdir(parents=True)
        bars = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
                "symbol": resolved,
                "timeframe": "M1",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "tick_volume": 10.0,
                "spread": 20.0,
                "real_volume": 0.0,
            }
        )[BAR_COLUMNS]
        bars.to_parquet(base / "d.parquet", index=False)

    # XAUUSDm (shorter) is preferred over XAUUSD247m.
    files = tasks.find_collected_bars("XAUUSD", "M1", root=tmp_path)
    assert len(files) == 1
    assert "XAUUSDm" in str(files[0])


def test_align_hybrid_mt5_bars_exness_ticks(tmp_path: Path, monkeypatch) -> None:
    from slytrade.data.schemas import BAR_COLUMNS, TICK_COLUMNS

    # MT5 bars under resolved symbol.
    bar_dir = tmp_path / "mt5_bars" / "symbol=XAUUSDm" / "timeframe=M1" / "y=2026"
    bar_dir.mkdir(parents=True)
    bar_times = pd.date_range("2026-01-01", periods=120, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {
            "time": bar_times,
            "symbol": "XAUUSDm",
            "timeframe": "M1",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "tick_volume": 10.0,
            "spread": 20.0,
            "real_volume": 0.0,
        }
    )[BAR_COLUMNS]
    bars.to_parquet(bar_dir / "d.parquet", index=False)

    # Exness ticks under base symbol (canonical Exness layout).
    tick_dir = tmp_path / "exness_ticks" / "symbol=XAUUSD" / "year=2026" / "month=01"
    tick_dir.mkdir(parents=True)
    tick_times = pd.date_range("2026-01-01", periods=1200, freq="6s", tz="UTC")
    mid = 100.0 + pd.Series(range(1200), dtype=float) * 0.0001
    ticks = pd.DataFrame(
        {
            "time_msc": tick_times,
            "time": tick_times.floor("s"),
            "symbol": "XAUUSD",
            "bid": (mid - 0.01).round(3),
            "ask": (mid + 0.01).round(3),
            "last": 0.0,
            "volume": 1.0,
            "volume_real": 0.0,
            "flags": 0.0,
            "spread": 0.02,
            "mid": mid,
        }
    )[TICK_COLUMNS]
    ticks.to_parquet(tick_dir / "period=2026-01.parquet", index=False)

    monkeypatch.setattr(tasks, "SAMPLE_ROOT", str(tmp_path / "samples"))
    monkeypatch.setattr(tasks, "EXNESS_DERIVED_ROOT", str(tmp_path / "exness_derived"))
    monkeypatch.chdir(tmp_path)

    result = tasks.align("XAUUSD", timeframe="M1", root=str(tmp_path))
    assert result.ok, result.message
    # Hybrid alignment: canonical symbol XAUUSD, MT5 bars + Exness ticks.
    assert result.data is not None


def test_merge_tick_sources_writes_merged_set(tmp_path: Path, monkeypatch) -> None:
    """Exness history + MT5 recent ticks merge into one deduplicated tick set."""
    from slytrade.data.schemas import TICK_COLUMNS

    monkeypatch.setattr(tasks, "SAMPLE_ROOT", str(tmp_path / "samples"))
    monkeypatch.setattr(tasks, "EXNESS_DERIVED_ROOT", str(tmp_path / "exness_derived"))

    def fake_download(symbol, *, lookback, root):
        return tasks.TaskResult(True, "ok", {"ticks": 2})

    monkeypatch.setattr(tasks, "_download_exness_ticks", fake_download)

    def fake_mt5_ticks(symbol, *, lookback, root):
        return tasks.TaskResult(True, "ok", {"ticks": 1})

    monkeypatch.setattr(tasks, "_collect_ticks_from_mt5", fake_mt5_ticks)

    exness = pd.DataFrame(
        {
            "time_msc": pd.date_range("2026-08-10", periods=3, freq="s", tz="UTC"),
            "time": pd.date_range("2026-08-10", periods=3, freq="s", tz="UTC").floor("s"),
            "symbol": "XAUUSD",
            "bid": 100.0,
            "ask": 100.02,
            "last": 0.0,
            "volume": 1.0,
            "volume_real": 0.0,
            "flags": 0.0,
            "spread": 0.02,
            "mid": 100.01,
        }
    )[TICK_COLUMNS]
    recent = pd.DataFrame(
        {
            "time_msc": pd.date_range("2026-08-14", periods=2, freq="s", tz="UTC"),
            "time": pd.date_range("2026-08-14", periods=2, freq="s", tz="UTC").floor("s"),
            "symbol": "XAUUSDm",
            "bid": 101.0,
            "ask": 101.02,
            "last": 0.0,
            "volume": 1.0,
            "volume_real": 0.0,
            "flags": 0.0,
            "spread": 0.02,
            "mid": 101.01,
        }
    )[TICK_COLUMNS]

    monkeypatch.setattr(tasks, "load_exness_ticks", lambda symbol, root=None: exness.copy())
    monkeypatch.setattr(tasks, "load_collected_ticks", lambda symbol, root=None: recent.copy())

    result = tasks._merge_tick_sources("XAUUSD", lookback="1m", root=tmp_path, recent_days=3)
    assert result.ok, result.message
    assert result.data["ticks"] == 5  # 3 exness + 2 recent, no dup
    merged = tasks.load_merged_ticks("XAUUSD", root=tmp_path)
    assert len(merged) == 5
    # All relabeled to the canonical base symbol.
    assert set(merged["symbol"].unique()) == {"XAUUSD"}


def test_align_prefers_merged_ticks(tmp_path: Path, monkeypatch) -> None:
    from slytrade.data.schemas import BAR_COLUMNS, TICK_COLUMNS

    monkeypatch.setattr(tasks, "SAMPLE_ROOT", str(tmp_path / "samples"))
    monkeypatch.setattr(tasks, "EXNESS_DERIVED_ROOT", str(tmp_path / "exness_derived"))

    bar_dir = tmp_path / "mt5_bars" / "symbol=XAUUSDm" / "timeframe=M1" / "y=2026"
    bar_dir.mkdir(parents=True)
    bar_times = pd.date_range("2026-01-01", periods=120, freq="min", tz="UTC")
    bars = pd.DataFrame(
        {
            "time": bar_times,
            "symbol": "XAUUSDm",
            "timeframe": "M1",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "tick_volume": 10.0,
            "spread": 20.0,
            "real_volume": 0.0,
        }
    )[BAR_COLUMNS]
    bars.to_parquet(bar_dir / "d.parquet", index=False)

    # Both an Exness-only and a MERGED tick set exist; align must use merged.
    for subdir, count, sym in (("exness_ticks", 1000, "XAUUSD"), ("merged_ticks", 2000, "XAUUSD")):
        base = tmp_path / subdir / f"symbol={sym}" / "year=2026" / "month=01"
        base.mkdir(parents=True)
        times = pd.date_range("2026-01-01", periods=count, freq="6s", tz="UTC")
        mid = 100.0 + pd.Series(range(count), dtype=float) * 0.0001
        ticks = pd.DataFrame(
            {
                "time_msc": times,
                "time": times.floor("s"),
                "symbol": sym,
                "bid": (mid - 0.01).round(3),
                "ask": (mid + 0.01).round(3),
                "last": 0.0,
                "volume": 1.0,
                "volume_real": 0.0,
                "flags": 0.0,
                "spread": 0.02,
                "mid": mid,
            }
        )[TICK_COLUMNS]
        ticks.to_parquet(base / "p.parquet", index=False)

    monkeypatch.chdir(tmp_path)
    result = tasks.align("XAUUSD", timeframe="M1", root=str(tmp_path))
    assert result.ok, result.message
    import json

    manifest = json.loads((tmp_path / "data" / "processed" / "aligned" / "XAUUSD" / "manifest.json").read_text())
    assert manifest["tick_source"] == "merged_ticks"


def test_with_reward_applies_trade_pnl() -> None:
    config = RLEnvironmentConfig(reward_type="raw")
    adjusted = tasks._with_reward(config, "trade_pnl")
    assert adjusted.reward_type == "trade_pnl"


def test_with_reward_rejects_unknown() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown reward type"):
        tasks._with_reward(RLEnvironmentConfig(), "bogus")


def test_samples_source_does_not_require_data_raw(tmp_path: Path, monkeypatch) -> None:
    """The samples source writes to SAMPLE_ROOT, so it must not require the
    broker data root (data/raw) to exist — e.g. after the operator deleted it."""
    checked: list[str] = []

    def spy(root):
        checked.append(str(root))
        return None

    monkeypatch.setattr(tasks, "SAMPLE_ROOT", str(tmp_path / "samples"))
    monkeypatch.setattr(tasks, "_ensure_writable_root", spy)
    monkeypatch.setattr(tasks, "_ensure_standard_dirs", lambda: None)

    result = tasks.collect_all("XAUUSD", source="samples", sample_start="2025-01-01")
    assert result.ok
    assert all("data/raw" not in c for c in checked)
    assert any(str(tmp_path / "samples") in c for c in checked)


def test_ensure_standard_dirs_recreates_deleted_tree(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "data").exists()
    error = tasks._ensure_standard_dirs()
    assert error is None
    for rel in ("data/raw", "data/processed", "data/exness_derived", "data/samples", "models", "logs", "state"):
        assert (tmp_path / rel).is_dir(), rel
