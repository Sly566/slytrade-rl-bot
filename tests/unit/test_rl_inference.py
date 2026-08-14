from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.execution.models import Side
from slytrade.rl.inference import RLPolicyStrategy


class FixedActionModel:
    def __init__(self, action: int) -> None:
        self.action = action

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return int(self.action), None


def make_bars(n: int = 60) -> pd.DataFrame:
    times = pd.date_range("2026-08-14T10:00:00", periods=n, freq="min", tz="UTC")
    close = 100.0 + pd.Series(range(n), dtype=float) * 0.01
    return pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",
            "open": close - 0.005,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "tick_volume": 100.0,
        }
    )


def make_strategy(action: int) -> RLPolicyStrategy:
    from slytrade.ml.features import ML_FEATURE_COLUMNS, fit_scaler

    bars = make_bars()
    features = pd.DataFrame(index=bars.index)
    for column in ML_FEATURE_COLUMNS:
        features[column] = np.linspace(0.0, 1.0, len(bars))
    scaler = fit_scaler(features)
    return RLPolicyStrategy(
        model=FixedActionModel(action),
        feature_columns=tuple(ML_FEATURE_COLUMNS),
        scaler_params=scaler,
        symbol="XAUUSD",
    )


def test_long_action_emits_buy() -> None:
    strategy = make_strategy(1)
    intent = None
    for index, bar in make_bars(40).iterrows():
        intent = strategy.on_bar(index, bar)
        if intent is not None:
            break
    assert intent is not None
    assert intent.side == Side.BUY
    assert intent.volume > 0


def test_short_action_emits_sell() -> None:
    strategy = make_strategy(2)
    intent = None
    for index, bar in make_bars(40).iterrows():
        intent = strategy.on_bar(index, bar)
        if intent is not None:
            break
    assert intent is not None
    assert intent.side == Side.SELL


def test_hold_action_emits_nothing() -> None:
    strategy = make_strategy(0)
    for index, bar in make_bars(40).iterrows():
        assert strategy.on_bar(index, bar) is None


def test_flatten_after_long_emits_sell() -> None:
    # Enter long first, then flatten.
    enter = make_strategy(1)
    entry_intent = None
    for index, bar in make_bars(40).iterrows():
        entry_intent = enter.on_bar(index, bar)
        if entry_intent is not None:
            break
    assert entry_intent is not None

    strategy = make_strategy(3)
    strategy._side = "long"
    intent = None
    for index, bar in make_bars(40).iterrows():
        intent = strategy.on_bar(index, bar)
        if intent is not None:
            break
    assert intent is not None
    assert intent.side == Side.SELL
    assert "rl_flatten" in intent.reason


def test_minimum_history_window() -> None:
    strategy = make_strategy(1)
    bars = make_bars(40)
    # Too little history -> no signal yet.
    assert strategy.on_bar(0, bars.iloc[0]) is None
    assert strategy.on_bar(1, bars.iloc[1]) is None
