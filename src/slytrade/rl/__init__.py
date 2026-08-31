"""SlyTrade RL — Reinforcement Learning layer for signal filtering + exit optimization.

Uses Gymnasium + Stable-Baselines3 to train an agent that learns:
- Which signals to take vs skip (signal filter)
- When to close positions (exit optimization)
- How to size positions (dynamic risk)

Architecture:
  1. SlyTradeEnv(gym.Env) — wraps backtest engine as standard RL env
  2. Observation: market features + position state + account state
  3. Action: discrete (hold/close/enter_long/enter_short) + continuous (size/SL/TP)
  4. Reward: risk-adjusted returns with drawdown penalty
  5. Training: PPO/SAC with Optuna hyperparameter tuning
"""

from .env import SlyTradeEnv
from .reward import RewardConfig, compute_reward
from .train import train_agent

__all__ = ["SlyTradeEnv", "RewardConfig", "compute_reward", "train_agent"]
