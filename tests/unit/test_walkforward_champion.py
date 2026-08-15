"""Tests for the multi-fold walk-forward + champion comparison."""

from __future__ import annotations

import pandas as pd
import pytest

from slytrade.tasks import _add_champion_comparison


def _bars(n: int = 2000) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    close = 100.0 + pd.Series(range(n), dtype=float) * 0.001
    return pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSDm",
            "timeframe": "M1",
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
        }
    )


class _Fold:
    def __init__(self, index, test_start, test_end):
        self.index = index
        self.test_start = test_start
        self.test_end = test_end


def test_champion_comparison_adds_columns(tmp_path, monkeypatch) -> None:
    pytest.importorskip("gymnasium")

    from slytrade import tasks

    # Build an aligned dataset so the champion backtest has ICT features.
    tasks.SAMPLE_ROOT = str(tmp_path / "samples")
    tasks.generate_sample_dataset("XAUUSD", start="2025-01-01", bar_periods=4000, tick_periods=40000, out_dir=str(tmp_path / "samples"))
    aligned = tasks.align("XAUUSD", timeframe="M1", out_dir=str(tmp_path / "aligned"))
    bars = pd.read_parquet(aligned.data["bars_file"])

    from slytrade.rl.dataset import build_rl_dataset

    dataset = build_rl_dataset(bars)
    folds = [_Fold(0, 3000, 4000)]
    table = pd.DataFrame(
        {
            "fold": [0, "AGGREGATE"],
            "test_mean_total_return": [-0.01, -0.01],
        }
    )
    result = _add_champion_comparison(table, dataset, folds, "XAUUSD")
    assert "persona_return" in result.columns
    assert "rl_minus_persona" in result.columns
    # Champion return is a finite number (the strategy ran on the test slice).
    assert result.loc[0, "persona_return"] == result.loc[0, "persona_return"]  # not NaN


def test_walk_forward_multiple_folds(tmp_path, monkeypatch) -> None:
    """60k/15k/15k windows on a ~171k-bar dataset (the real 6m case) yield >1 fold."""
    from slytrade.rl.walkforward import make_walk_forward_folds, resolve_fold_windows

    total = 171_743
    windows = resolve_fold_windows(total, train_window=60_000, validation_window=15_000, test_window=15_000, embargo=500)
    folds = make_walk_forward_folds(
        total,
        train_window=windows.train_window,
        validation_window=windows.validation_window,
        test_window=windows.test_window,
        embargo=windows.embargo,
        step=windows.step,
    )
    assert len(folds) > 1, f"expected multiple folds, got {len(folds)}"
