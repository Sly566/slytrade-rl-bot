"""Regression tests for the superbrain (top-tier RL/ML) upgrades.

Locks in: the persona_action label is NOT in the observation (no circularity),
the 6-scalar agent state incl. cooldown, class-balanced BC + DAgger functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.rl.dataset import build_rl_dataset
from slytrade.rl.environment import RLEnvironmentConfig, SlyTradeRLEnvironment
from slytrade.rl.walkforward import behavioral_clone, dagger_refine, persona_actions_for_bars


def make_bars(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    bull = np.tile(np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0]), n // 6 + 1)[:n]
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
            "symbol": "XAUUSD", "timeframe": "M15",
            "open": close, "high": close + 0.4, "low": close - 0.4, "close": close,
            "tick_volume": 100.0, "spread": 2.0, "real_volume": 0.0,
            "atr": 0.4, "bos_dir": bull, "choch_dir": 0.0, "liquidity_sweep": -bull,
            "fvg_bullish": (bull > 0).astype(float), "fvg_bearish": (bull < 0).astype(float),
            "order_block_bullish": 0.0, "order_block_bearish": 0.0,
            "premium_discount": -0.5 * bull, "trend_strength": 0.5 * bull,
            "mtf_bias": bull, "mtf_confluence_score": 3.0,
        }
    )


def test_persona_action_is_not_in_rl_observation():
    bars = make_bars(400)
    dataset = build_rl_dataset(bars)
    assert "persona_action" not in dataset.features.columns
    # The legitimate summary signal IS present.
    assert "persona_score" in dataset.features.columns
    assert "persona_bias" in dataset.features.columns


def test_agent_state_has_six_scalars_including_cooldown():
    features = pd.DataFrame({"f": np.zeros(100, dtype=float)})
    bars = make_bars(100)
    env = SlyTradeRLEnvironment(features=features, bars=bars, config=RLEnvironmentConfig(reward_type="r_multiple"))
    obs, _ = env.reset()
    assert obs.shape == (1 + 6,)
    # cooldown remaining at reset (never entered -> cooldown fully elapsed = 0)
    assert obs[-1] == 0.0


def test_behavioral_clone_and_dagger_run():
    # Smoke test: the top-tier BC + DAgger path must run without error and
    # produce a finite policy. (No torch needed for the smoke — BC imports it.)
    import torch  # noqa: F401

    bars = make_bars(200)
    dataset = build_rl_dataset(bars)
    scaler = dataset.fit_scaler(0, len(bars))
    cfg = RLEnvironmentConfig(seed=42, reward_type="r_multiple", shaping_enabled=False,
                              stop_loss_atr=1.0, take_profit_atr=3.0, max_bars_in_trade=60,
                              episode_length_bars=0, round_trip_cost_r=0.043, entry_cooldown_bars=10)
    env = dataset.env_factory(0, len(bars), seed=42, scaler_params=scaler, config=cfg)
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", env, seed=42)
    actions = persona_actions_for_bars(bars)
    # collect demonstrations via the env
    from slytrade.rl.walkforward import collect_demonstrations

    obs, act = collect_demonstrations(env, actions)
    behavioral_clone(model, obs, act, epochs=2)
    dagger_refine(model, env, actions, epochs=1)
    # The cloned policy must produce valid actions (0..3).
    o, _ = env.reset()
    a, _ = model.predict(o, deterministic=True)
    assert int(a) in (0, 1, 2, 3)
