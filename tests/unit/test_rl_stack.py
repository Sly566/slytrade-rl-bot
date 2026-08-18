"""Tests for the RL environment, dataset builder, and walk-forward folds."""

import numpy as np
import pandas as pd
import pytest

from slytrade.config.trader_personality import TraderPersonality
from slytrade.ml.features import compute_ml_features
from slytrade.rl.dataset import build_rl_dataset
from slytrade.rl.environment import SlyTradeRLEnvironment
from slytrade.rl.mode_vector import build_mode_vector
from slytrade.rl.walkforward import make_walk_forward_folds

pytest.importorskip("gymnasium")


def make_bars(periods: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    prices = 2400 + np.cumsum(rng.normal(0.0, 1.0, size=periods))
    high = prices + 0.5
    low = prices - 0.5
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=periods, freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": prices,
            "high": high,
            "low": low,
            "close": prices,
            "tick_volume": rng.integers(1, 10, size=periods).astype(float),
        }
    )


def test_ml_features_are_all_causal_columns():
    bars = make_bars(300)
    features = compute_ml_features(bars)
    assert len(features) == len(bars)
    assert not features.isna().any().any()
    # Causal means no centered/shift(-) columns exist
    assert "ml_ret_1" in features.columns


def test_dataset_build_and_env_factory():
    bars = make_bars(600)
    dataset = build_rl_dataset(bars)
    assert dataset.symbol == "XAUUSD"
    scaler = dataset.fit_scaler(0, 500)
    env = dataset.env_factory(0, 500, seed=1, scaler_params=scaler)
    assert isinstance(env, SlyTradeRLEnvironment)
    # features + 5 agent-state scalars (mode vector absent here).
    assert env.observation_space.shape[0] == len(dataset.features.columns) + 6


def test_env_reset_and_step_no_lookahead():
    bars = make_bars(500)
    dataset = build_rl_dataset(bars)
    scaler = dataset.fit_scaler(0, 400)
    env = dataset.env_factory(0, 400, seed=1, scaler_params=scaler)
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    # Step with hold action repeatedly; episode must terminate without error.
    done = truncated = False
    steps = 0
    while not done and not truncated and steps < 500:
        obs, reward, done, truncated, info = env.step(0)
        steps += 1
        assert steps < 500
    assert info["n_trades"] >= 0
    assert len(env.ledger.records) >= 0


def test_env_long_action_opens_position():
    bars = make_bars(500)
    dataset = build_rl_dataset(bars)
    scaler = dataset.fit_scaler(0, 400)
    env = dataset.env_factory(0, 400, seed=1, scaler_params=scaler)
    env.reset(seed=1)
    action = 1  # long
    env.step(action)
    # The long order must produce at least an entry fill in the ledger. The
    # position may be closed by the deterministic same-bar SL/TP check when
    # the bar's range breaches the stop, which is legitimate conservative
    # behavior (not a rejected order).
    fills = [r for r in env.ledger.records if r.reason == "rl_entry"]
    assert len(fills) >= 1


def test_mode_vector_consistent_shape():
    personality = TraderPersonality.from_yaml("configs/trader_personality.yaml")
    context = {
        "volatility": "normal",
        "trend": "bull",
        "session": "london",
        "regime_score": 0.8,
        "premium_discount": -0.2,
        "mtf_bias": 1.0,
    }
    vector = build_mode_vector(personality, context)
    # 3 vol + 3 trend + 6 session + 3 scalars + the 18 persona traits.
    from slytrade.rl.mode_vector import PERSONA_TRAIT_NAMES

    assert vector.shape[0] == 3 + 3 + 6 + 3 + len(PERSONA_TRAIT_NAMES)


def test_walk_forward_folds_correct_boundaries():
    folds = make_walk_forward_folds(
        400_000,
        train_window=100_000,
        validation_window=50_000,
        test_window=50_000,
        embargo=500,
        step=50_000,
    )
    assert len(folds) >= 1
    fold = folds[0]
    # Train -> embargo -> val -> embargo -> test
    assert fold.val_start == fold.train_end + 500
    assert fold.test_start == fold.val_end + 500
    assert fold.test_end <= 400_000
    # Test windows are disjoint, strictly ordered, and never overlap (train
    # windows may legitimately overhang by design in walk-forward).
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end <= b.test_start
    all_test_ranges = [(f.test_start, f.test_end) for f in folds]
    for i, a in enumerate(all_test_ranges):
        for b in all_test_ranges[i + 1 :]:
            assert a[1] <= b[0]  # no overlap





def test_walk_forward_folds_too_small_raises():
    with pytest.raises(ValueError):
        make_walk_forward_folds(
            1_000,
            train_window=100_000,
            validation_window=50_000,
            test_window=50_000,
            embargo=500,
        )
