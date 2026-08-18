"""Tests for the RL environment v2: managed SL/TP exits + feature adoption."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from slytrade.rl.environment import RLEnvironmentConfig, SlyTradeRLEnvironment  # noqa: E402


def make_bars(n: int = 300, *, drift: float = 0.0) -> pd.DataFrame:
    times = pd.date_range("2026-08-14T10:00:00", periods=n, freq="min", tz="UTC")
    close = 100.0 + pd.Series(range(n), dtype=float) * drift
    atr = pd.Series(0.5, index=range(n))
    return pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "atr": atr,
        }
    )


def make_env(bars: pd.DataFrame, **config_kwargs) -> SlyTradeRLEnvironment:
    features = pd.DataFrame({"f1": np.zeros(len(bars), dtype=float), "f2": np.ones(len(bars), dtype=float)})
    kwargs = {"reward_type": "trade_pnl", "use_managed_exits": True, "episode_length_bars": 0}
    kwargs.update(config_kwargs)
    config = RLEnvironmentConfig(**kwargs)
    return SlyTradeRLEnvironment(features=features, bars=bars, config=config)


def test_managed_take_profit_exits_and_rewards() -> None:
    # Steadily rising price: a long should hit TP (2×ATR) and realize a profit.
    bars = make_bars(200, drift=0.1)
    env = make_env(bars, stop_loss_atr=1.0, take_profit_atr=2.0)
    env.reset()
    env.step(1)  # enter long
    total_reward = 0.0
    for _ in range(100):
        _, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        if env._position == 0:
            break
    assert env._position == 0  # position was closed by the managed exit
    assert total_reward > 0  # TP exit realizes a profit


def test_managed_stop_loss_exits_and_rewards() -> None:
    # Steadily falling price: a long should hit SL and realize a loss.
    bars = make_bars(200, drift=-0.1)
    env = make_env(bars)
    env.reset()
    env.step(1)  # enter long into a falling market
    total_reward = 0.0
    for _ in range(100):
        _, reward, terminated, truncated, info = env.step(0)
        total_reward += reward
        if env._position == 0:
            break
    assert env._position == 0
    assert total_reward < 0


def test_sparse_reward_while_holding_is_zero() -> None:
    # With managed exits on but no hit, holding rewards exactly zero.
    bars = make_bars(300, drift=0.001)
    env = make_env(bars)
    env.reset()
    env.step(1)
    for _ in range(10):
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0


def test_episode_end_flattens_open_position() -> None:
    bars = make_bars(50, drift=0.01)
    env = make_env(bars, episode_length_bars=10)
    env.reset()
    env.step(1)
    for _ in range(30):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert env._position == 0  # truncated episode must not leak an open position


def test_feature_columns_adopted_from_aligned_bars() -> None:
    from slytrade.rl.dataset import build_rl_dataset

    bars = make_bars(120, drift=0.01)
    # Simulate aligned bars that carry ICT/MTF/tick columns.
    bars["bos_dir"] = 1.0
    bars["premium_discount"] = 0.0
    bars["mtf_bias"] = 1.0
    bars["mtf_confluence_score"] = 3.0
    bars["session_london"] = 1.0
    bars["htf_h1_bos_dir"] = 1.0
    bars["tick_rate_per_second"] = 2.0
    bars["tick_mid_return"] = 0.001

    dataset = build_rl_dataset(bars)
    columns = set(dataset.features.columns)
    assert "bos_dir" in columns
    assert "mtf_bias" in columns
    assert "htf_h1_bos_dir" in columns
    assert "session_london" in columns
    assert "tick_rate_per_second" in columns
    # ML features are always present too.
    assert "ml_ret_1" in columns


def test_observation_includes_mode_vector_dimension() -> None:
    bars = make_bars(120, drift=0.01)
    features = pd.DataFrame({"f1": np.zeros(120, dtype=float)})
    config = RLEnvironmentConfig(reward_type="trade_pnl")
    env = SlyTradeRLEnvironment(features=features, bars=bars, config=config, mode_vector=np.zeros(6, dtype=np.float32))
    obs, _ = env.reset()
    # 1 feature + 6 mode + 5 agent-state scalars.
    assert obs.shape == (1 + 6 + 6,)


def test_observation_includes_agent_state() -> None:
    bars = make_bars(120, drift=0.01)
    features = pd.DataFrame({"f1": np.zeros(120, dtype=float)})
    config = RLEnvironmentConfig(reward_type="trade_pnl")
    env = SlyTradeRLEnvironment(features=features, bars=bars, config=config)
    obs, _ = env.reset()
    # The last 5 elements are the agent-state vector; position starts at 0.
    assert obs.shape == (1 + 6,)
    assert obs[-6] == 0.0  # position
    assert obs[-2] == 0.0  # episode progress at reset (step 0)
