"""Tests for the production R-multiple reward and its shaping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from slytrade.rl.environment import RLEnvironmentConfig, SlyTradeRLEnvironment  # noqa: E402
from slytrade.rl.rewards import opening_cost_r, r_from_fraction  # noqa: E402


def test_r_from_fraction_converts_to_r() -> None:
    # fraction = (exit - entry)/entry; stop_distance is the risk.
    # 2R trade: price up by 2× the stop distance.
    assert r_from_fraction(0.02, entry_price=100.0, stop_distance=1.0) == pytest.approx(2.0)
    assert r_from_fraction(-0.01, entry_price=100.0, stop_distance=1.0) == pytest.approx(-1.0)
    assert r_from_fraction(0.05, entry_price=0.0, stop_distance=1.0) == 0.0


def test_opening_cost_r() -> None:
    cost = opening_cost_r(0.0002, entry_price=100.0, stop_distance=1.0)
    assert cost == pytest.approx(0.02)  # 0.0002 * 100 / 1


def make_bars(n: int = 300, *, setup: bool = True) -> pd.DataFrame:
    times = pd.date_range("2026-08-14T10:00:00", periods=n, freq="min", tz="UTC")
    close = 100.0 + pd.Series(range(n), dtype=float) * 0.01
    bars = pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "atr": 0.5,
            "bos_dir": 1.0 if setup else 0.0,
            "choch_dir": 0.0,
            "liquidity_sweep": -1.0 if setup else 0.0,
            "fvg_bullish": 1.0 if setup else 0.0,
            "order_block_bullish": 0.0,
            "premium_discount": -0.2 if setup else 0.0,
            "trend_strength": 0.4 if setup else 0.0,
        }
    )
    return bars


def make_env(bars: pd.DataFrame, **kwargs) -> SlyTradeRLEnvironment:
    features = pd.DataFrame({"f1": np.zeros(len(bars), dtype=float), "f2": np.ones(len(bars), dtype=float)})
    config = RLEnvironmentConfig(
        reward_type="r_multiple", use_managed_exits=True, episode_length_bars=0, **kwargs
    )
    return SlyTradeRLEnvironment(features=features, bars=bars, config=config)


def test_setup_score_mirrors_ict_confluence() -> None:
    bars = make_bars(setup=True)
    env = make_env(bars)
    assert env._setup_score(bars.iloc[5]) >= env.config.setup_score_threshold
    env2 = make_env(make_bars(setup=False))
    assert env2._setup_score(make_bars(setup=False).iloc[5]) == 0


def test_r_multiple_rewards_managed_tp() -> None:
    # Rising market, long entry → TP (2×ATR) hit → reward ≈ +2R minus costs.
    bars = make_bars(400, setup=True)
    env = make_env(bars)
    env.reset()
    env.step(1)  # enter long
    total = 0.0
    for _ in range(300):
        _, reward, terminated, truncated, _ = env.step(0)
        total += reward
        if env._position == 0:
            break
    assert env._position == 0
    # The managed TP exit pays ~2R, so the net is clearly positive.
    assert total > 1.0


def test_r_multiple_charges_regret_when_flat_on_setup() -> None:
    bars = make_bars(60, setup=True)
    env = make_env(bars)
    env.reset()
    # Stay flat on a high-confluence setup → regret charges once.
    rewards = []
    for _ in range(5):
        _, reward, _, _, _ = env.step(0)
        rewards.append(reward)
    assert any(r < 0 for r in rewards)
    assert sum(rewards) < 0


def test_r_multiple_no_regret_without_setup() -> None:
    bars = make_bars(60, setup=False)
    env = make_env(bars)
    env.reset()
    for _ in range(5):
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0


def test_r_multiple_entry_quality_bonus_on_high_setup() -> None:
    bars = make_bars(200, setup=True)
    env = make_env(bars)
    env.reset()
    _, reward, _, _, _ = env.step(1)  # enter long on a high-confluence setup
    # Entry shaping: bonus for quality setup (minus opening cost) > 0.
    assert reward > 0
