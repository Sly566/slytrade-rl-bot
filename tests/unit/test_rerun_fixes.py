"""Tests for the two re-run bugs: dataset fragmentation warnings and the
deterministic model-id registry collision."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.rl.dataset import build_rl_dataset
from slytrade.rl.mode_vector import mode_matrix_columns


def _bars(n: int = 400) -> pd.DataFrame:
    times = pd.date_range("2026-08-14T08:00:00", periods=n, freq="min", tz="UTC")
    close = 100.0 + pd.Series(range(n), dtype=float) * 0.01
    bars = pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
        }
    )
    # Simulate the aligned bars' rich column set: many ICT/MTF/tick columns.
    for i in range(40):
        bars[f"htf_h1_f{i}"] = np.linspace(0.0, 1.0, n)
    for column in ("bos_dir", "choch_dir", "premium_discount", "liquidity_sweep", "atr", "trend_strength", "mtf_bias", "mtf_confluence_score", "tick_rate_per_second", "session_london"):
        bars[column] = np.zeros(n, dtype=float)
    return bars


def test_dataset_build_emits_no_fragmentation_warning() -> None:
    bars = _bars()
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        dataset = build_rl_dataset(bars, TraderPersonality())
    assert len(dataset.features) == len(bars)
    # Constant columns (fixed persona traits, all-zero synthetic flags) are
    # dropped by design — they carry no signal. Varying columns must survive.
    for column in mode_matrix_columns():
        if column.startswith("mode_p_"):
            assert column not in dataset.features.columns, column
    assert "htf_h1_f0" in dataset.features.columns


def test_dataset_adopts_all_htf_columns() -> None:
    bars = _bars()
    dataset = build_rl_dataset(bars, TraderPersonality())
    htf = [column for column in dataset.features.columns if column.startswith("htf_")]
    assert len(htf) >= 40


def test_train_model_ids_are_unique_across_runs(tmp_path) -> None:
    """Re-running train must register a new artifact, not collide."""
    import pytest

    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")

    from slytrade import tasks

    # Build a small aligned bars file.
    tasks.SAMPLE_ROOT = str(tmp_path / "samples")
    tasks.generate_sample_dataset("XAUUSD", start="2025-01-01", bar_periods=3000, tick_periods=30000, out_dir=str(tmp_path / "samples"))
    aligned = tasks.align("XAUUSD", timeframe="M1", out_dir=str(tmp_path / "aligned"))
    bars_file = aligned.data["bars_file"]

    first = tasks.train(bars_file, symbol="XAUUSD", total_timesteps=300, artifacts_dir=str(tmp_path / "artifacts"), registry_path=str(tmp_path / "registry.jsonl"))
    assert first.ok, first.message
    second = tasks.train(bars_file, symbol="XAUUSD", total_timesteps=300, artifacts_dir=str(tmp_path / "artifacts"), registry_path=str(tmp_path / "registry.jsonl"))
    assert second.ok, second.message
    assert first.data["model_id"] != second.data["model_id"]
