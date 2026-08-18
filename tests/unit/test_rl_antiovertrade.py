"""Regression tests for the anti-overtrading RL upgrades.

Locked in from the real-data diagnosis: the r_multiple shaping (entry bonus /
missed-setup regret) taught the agent to overtrade ~150 trades/episode and lose
~0.5R/trade out-of-sample. The fixes:
- shaping OFF by default (pure realised R minus opening cost),
- activity brake (max_trades_per_episode),
- the persona confluence signal as observation features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.rl.dataset import persona_signal_columns
from slytrade.rl.environment import RLEnvironmentConfig, SlyTradeRLEnvironment


def make_bars(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    bull = np.tile(np.array([1.0, -1.0]), n // 2 + 1)[:n]
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "symbol": "XAUUSD", "timeframe": "M1",
            "open": close, "high": close + 0.2, "low": close - 0.2, "close": close,
            "atr": 0.2, "bos_dir": bull, "choch_dir": 0.0, "liquidity_sweep": -bull,
            "fvg_bullish": (bull > 0).astype(float), "fvg_bearish": (bull < 0).astype(float),
            "order_block_bullish": 0.0, "order_block_bearish": 0.0,
            "premium_discount": -0.5 * bull, "trend_strength": 0.5 * bull,
        }
    )


def test_shaping_defaults_off():
    cfg = RLEnvironmentConfig(reward_type="r_multiple")
    assert cfg.shaping_enabled is False
    assert cfg.max_trades_per_episode == 0


def test_persona_signal_columns():
    bars = make_bars(200)
    sig = persona_signal_columns(bars)
    assert set(sig.columns) == {"persona_score", "persona_bias"}
    assert sig["persona_score"].between(0, 8).all()
    assert set(np.unique(sig["persona_bias"])).issubset({-1.0, 0.0, 1.0})


def test_activity_brake_caps_entries():
    features = pd.DataFrame({"f": np.zeros(200, dtype=float)})
    bars = make_bars(200)
    env = SlyTradeRLEnvironment(
        features=features, bars=bars,
        config=RLEnvironmentConfig(
            reward_type="r_multiple", use_managed_exits=True,
            episode_length_bars=0, max_trades_per_episode=3,
        ),
    )
    env.reset()
    # Drive entries explicitly: enter long, flatten, repeat — past the cap of
    # 3 the brake clamps the action to hold, so entries can never exceed 3.
    for _ in range(10):
        env.step(1)  # long
        env.step(3)  # flatten
    assert env._entries_this_episode <= 3
