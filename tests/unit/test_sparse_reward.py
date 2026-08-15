from __future__ import annotations

import pandas as pd
import pytest

from slytrade.rl.environment import RLEnvironmentConfig

gym = pytest.importorskip("gymnasium")

from slytrade.rl.environment import SlyTradeRLEnvironment  # noqa: E402


def make_env(reward_type: str = "trade_pnl") -> SlyTradeRLEnvironment:
    import numpy as np

    times = pd.date_range("2026-08-14T10:00:00", periods=200, freq="min", tz="UTC")
    close = 100.0 + np.linspace(0.0, 2.0, 200)
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
    features = pd.DataFrame({"f1": np.zeros(200, dtype=float), "f2": np.zeros(200, dtype=float)})
    config = RLEnvironmentConfig(reward_type=reward_type, transaction_cost=0.0002)
    return SlyTradeRLEnvironment(features=features, bars=bars, config=config)


def test_trade_pnl_is_sparse_while_holding() -> None:
    env = make_env("trade_pnl")
    env.reset()
    # Enter long.
    _, reward, _, _, _ = env.step(1)
    assert reward < 0  # opening costs transaction cost
    # Holding yields exactly zero reward.
    for _ in range(5):
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0


def test_trade_pnl_rewards_profitable_round_trip() -> None:
    env = make_env("trade_pnl")
    env.reset()
    env.step(1)  # enter long
    # Hold while price rises, then flatten.
    for _ in range(10):
        env.step(0)
    _, reward, _, _, _ = env.step(3)  # flatten
    # Price rose over the hold, so the realized PnL is positive.
    assert reward > 0


def test_trade_pnl_penalizes_losing_round_trip() -> None:
    import numpy as np

    times = pd.date_range("2026-08-14T10:00:00", periods=200, freq="min", tz="UTC")
    close = 100.0 - np.linspace(0.0, 2.0, 200)  # falling price
    bars = pd.DataFrame(
        {"time": times, "symbol": "XAUUSD", "open": close, "high": close + 0.1, "low": close - 0.1, "close": close}
    )
    features = pd.DataFrame({"f1": np.zeros(200, dtype=float)})
    env = SlyTradeRLEnvironment(
        features=features,
        bars=bars,
        # Managed exits disabled so the test isolates the action-driven
        # flatten path (the managed SL path is covered in test_rl_managed_exits).
        config=RLEnvironmentConfig(reward_type="trade_pnl", use_managed_exits=False),
    )
    env.reset()
    env.step(1)  # long into a falling market
    for _ in range(10):
        env.step(0)
    _, reward, _, _, _ = env.step(3)  # flatten
    assert reward < 0


def test_raw_reward_is_dense() -> None:
    env = make_env("raw")
    env.reset()
    env.step(1)  # enter
    # Raw reward is non-zero while holding (mark-to-market).
    nonzero = 0
    for _ in range(5):
        _, reward, _, _, _ = env.step(0)
        if reward != 0.0:
            nonzero += 1
    assert nonzero > 0


def test_entry_price_reset_on_reset() -> None:
    env = make_env("trade_pnl")
    env.reset()
    env.step(1)
    env.reset()
    assert env._entry_price == 0.0
    assert env._position == 0
