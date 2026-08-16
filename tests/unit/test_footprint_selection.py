"""Tests for the market-footprint objectives and dynamic feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.ml.feature_selection import (
    DynamicSelectionResult,
    apply_dynamic_selection,
    select_features_dynamic,
)
from slytrade.ml.footprint import (
    structure_direction,
    structure_r_objective,
    sweep_reversal_objective,
)


def make_bars(n: int = 600, *, sweep: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    times = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    atr = np.full(n, 0.5)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, n))
    high = close + 0.3
    low = close - 0.3
    bars = pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "atr": atr,
            "bos_dir": np.zeros(n),
            "choch_dir": np.zeros(n),
            "liquidity_sweep": np.zeros(n),
            "fvg_bullish": np.zeros(n),
            "order_block_bullish": np.zeros(n),
            "premium_discount": np.zeros(n),
            "trend_strength": np.zeros(n),
        }
    )
    # A block of bullish structure + sweeps.
    bars.loc[100:300, "bos_dir"] = 1.0
    bars.loc[100:300, "trend_strength"] = 0.5
    bars.loc[150:200, "liquidity_sweep"] = -1.0  # swept lows
    return bars


def test_structure_direction_signs() -> None:
    bars = make_bars()
    direction = structure_direction(bars)
    assert set(np.unique(direction)).issubset({-1.0, 0.0, 1.0})
    # The bullish block should bias positive.
    assert direction[150:300].mean() > 0.0


def test_structure_r_objective_is_causal_and_shaped() -> None:
    bars = make_bars()
    objective = structure_r_objective(bars, horizon=100)
    assert len(objective) == len(bars)
    assert pd.api.types.is_numeric_dtype(objective)
    # The last `horizon` bars have no forward window -> zero label.
    assert (objective.iloc[-100:] == 0.0).all()


def test_sweep_reversal_objective_detects_reversal() -> None:
    bars = make_bars(n=800)
    # Force a clean sweep-and-reverse: sweep lows at bar 200, then a strong
    # rally over the next 50 bars.
    bars.loc[200, "liquidity_sweep"] = -1.0
    bars.loc[201:250, "close"] = bars.loc[200, "close"] + np.arange(1, 51) * 0.1
    bars.loc[201:250, "high"] = bars.loc[201:250, "close"] + 0.1
    bars.loc[201:250, "low"] = bars.loc[201:250, "close"] - 0.1
    objective = sweep_reversal_objective(bars, horizon=50, atr_mult=1.0)
    assert objective.iloc[200] == 1.0


def _feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(index=bars.index)
    for i in range(12):
        frame[f"f{i}"] = rng.normal(0.0, 1.0, len(bars))
    # A genuinely predictive feature correlated with the structural direction.
    direction = structure_direction(bars)
    frame["signal"] = direction + rng.normal(0.0, 0.05, len(bars))
    return frame


def test_dynamic_selection_is_threshold_free_and_picks_signal() -> None:
    bars = make_bars()
    features = _feature_frame(bars)
    # Objectives are pre-aligned to the training slice (0:400).
    objectives = {
        "structure_r": structure_r_objective(bars, horizon=100).iloc[0:400].reset_index(drop=True),
        "sweep_reversal": sweep_reversal_objective(bars, horizon=100).iloc[0:400].reset_index(drop=True),
    }
    selection = select_features_dynamic(features, objectives, train_start=0, train_end=400)
    assert isinstance(selection, DynamicSelectionResult)
    # The count emerged from significance — not a hardcoded number.
    assert 0 < len(selection.selected) <= len(features.columns)
    # The crafted signal must be selected (it is genuinely predictive).
    assert "signal" in selection.selected


def test_dynamic_selection_apply() -> None:
    bars = make_bars()
    features = _feature_frame(bars)
    objectives = {"structure_r": structure_r_objective(bars, horizon=100).iloc[0:400].reset_index(drop=True)}
    selection = select_features_dynamic(features, objectives, train_start=0, train_end=400)
    reduced = apply_dynamic_selection(features, selection)
    assert list(reduced.columns) == list(selection.selected)


def test_dynamic_selection_falls_back_to_full_when_no_signal() -> None:
    rng = np.random.default_rng(3)
    n = 300
    features = pd.DataFrame({f"noise{i}": rng.normal(0.0, 1.0, n) for i in range(8)})
    objective = pd.Series(rng.normal(0.0, 1.0, n))  # pure noise target
    selection = select_features_dynamic(features, {"noise": objective}, train_start=0, train_end=n)
    # No feature beats the shadows -> full set returned (never blocks a run).
    assert len(selection.selected) == len(features.columns)
    assert selection.significant == ()


def test_dynamic_selection_rejects_bad_config() -> None:
    import pytest

    features = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    objectives = {"o": pd.Series([0.0, 1.0, 0.0])}
    with pytest.raises(ValueError):
        select_features_dynamic(features, objectives, train_start=0, train_end=10)
    with pytest.raises(ValueError):
        select_features_dynamic(features, {}, train_start=0, train_end=3)
