"""Regression tests for the RL environment's cost model and state features.

The old env charged transaction_cost as a price fraction (~0.0002), which
converted to ~0.08R/side = ~0.16R round trip on gold — ~4x the real M15 cost
(0.043R). That made even the champion barely break even inside the env, which
is why the RL could never learn the edge the backtest shows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.rl.environment import RLEnvironmentConfig, SlyTradeRLEnvironment


def make_bars(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
            "symbol": "XAUUSD", "timeframe": "M15",
            "open": close, "high": close + 0.3, "low": close - 0.3, "close": close,
            "atr": 0.5, "bos_dir": 1.0, "choch_dir": 0.0, "liquidity_sweep": -1.0,
            "fvg_bullish": 1.0, "fvg_bearish": 0.0, "order_block_bullish": 1.0,
            "order_block_bearish": 0.0, "premium_discount": -0.5, "trend_strength": 0.5,
        }
    )


def test_round_trip_cost_is_charged_once_per_entry():
    features = pd.DataFrame({"f": np.zeros(100, dtype=float)})
    bars = make_bars(100)
    env = SlyTradeRLEnvironment(
        features=features, bars=bars,
        config=RLEnvironmentConfig(reward_type="r_multiple", use_managed_exits=True,
                                   episode_length_bars=0, round_trip_cost_r=0.05, shaping_enabled=False),
    )
    env.reset()
    _, reward, _, _, _ = env.step(1)  # enter long on a setup bar
    # The opening reward is exactly the round-trip cost (no realised R yet).
    assert reward == -0.05


def test_agent_state_vector_present_and_sized():
    features = pd.DataFrame({"f": np.zeros(100, dtype=float)})
    bars = make_bars(100)
    env = SlyTradeRLEnvironment(features=features, bars=bars, config=RLEnvironmentConfig(reward_type="r_multiple"))
    obs, _ = env.reset()
    # 1 feature + 6 agent-state scalars.
    assert obs.shape == (1 + 6,)
    assert env.observation_space.shape[0] == 1 + 6


def test_candidate_mask_forces_hold_off_candidate_bars():
    features = pd.DataFrame({"f": np.zeros(100, dtype=float)})
    bars = make_bars(100)
    mask = np.zeros(100, dtype=np.float32)
    mask[10] = 2.0  # only bar 10 is a short candidate
    env = SlyTradeRLEnvironment(
        features=features, bars=bars,
        config=RLEnvironmentConfig(reward_type="r_multiple", use_managed_exits=False,
                                   episode_length_bars=0, mask_to_candidates=True),
        candidate_mask=mask,
    )
    env.reset()
    env.step(1)  # try to go long on bar 0 (not a candidate) -> must be ignored
    assert env._position == 0
