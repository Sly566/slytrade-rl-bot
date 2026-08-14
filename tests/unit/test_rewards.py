from __future__ import annotations

from slytrade.rl.rewards import RewardConfig, shaped_reward, sharpe_of_returns


def test_profit_reward_positive() -> None:
    reward = shaped_reward(previous_equity=1000.0, equity=1010.0, position=1, target_position=1, peak_equity=1010.0)
    assert reward > 0


def test_drawdown_penalty_applied() -> None:
    config = RewardConfig(return_weight=1.0, drawdown_penalty_weight=1.0, drawdown_tolerance=0.05)
    reward = shaped_reward(previous_equity=1000.0, equity=900.0, position=1, target_position=1, peak_equity=1100.0, config=config)
    assert reward < 0


def test_turnover_penalty_applied() -> None:
    flat = shaped_reward(previous_equity=1000.0, equity=1000.0, position=0, target_position=0, peak_equity=1000.0)
    turnover = shaped_reward(previous_equity=1000.0, equity=1000.0, position=0, target_position=1, peak_equity=1000.0)
    assert turnover < flat


def test_reward_is_clipped() -> None:
    config = RewardConfig(clip=1.0)
    reward = shaped_reward(previous_equity=1000.0, equity=2000.0, position=0, target_position=0, peak_equity=2000.0, config=config)
    assert reward == 1.0


def test_sharpe_of_returns() -> None:
    assert sharpe_of_returns([0.01, 0.02, 0.015]) > 0
    assert sharpe_of_returns([]) == 0.0
    assert sharpe_of_returns([0.01]) == 0.0
    assert sharpe_of_returns([0.0, 0.0]) == 0.0  # zero variance -> 0.0
    assert sharpe_of_returns([0.02, 0.01, 0.03, 0.015]) > 0
