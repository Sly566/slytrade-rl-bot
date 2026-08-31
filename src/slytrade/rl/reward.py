"""Reward function for SlyTrade RL agent.

The reward signal is the most critical part of RL training. A bad reward
function leads to degenerate behavior (e.g., never trading, or over-trading).

Reward Components:
  1. P&L component: realized profit/loss per trade (scaled)
  2. Risk-adjusted component: Sharpe-like ratio over recent window
  3. Drawdown penalty: heavy penalty for exceeding drawdown threshold
  4. Consistency bonus: reward for consecutive wins
  5. Activity penalty: small penalty for holding positions too long
  6. Opportunity cost: penalty for missing strong signals while flat

The reward is designed to produce a consistently profitable agent that:
- Takes high-probability setups (not every signal)
- Manages risk (small losses, big wins)
- Avoids large drawdowns
- Trades regularly (not idle for hours)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    # P&L scaling
    pnl_scale: float = 100.0          # multiply P&L fraction by this
    win_bonus: float = 0.1             # flat bonus for winning trade
    loss_penalty: float = 0.1          # flat penalty for losing trade

    # Risk-adjusted
    sharpe_window: int = 20            # trades window for rolling Sharpe
    sharpe_bonus_scale: float = 0.05   # bonus per unit of rolling Sharpe

    # Drawdown
    dd_threshold: float = 0.05         # 5% drawdown before penalty kicks in
    dd_penalty_scale: float = 0.5      # penalty = dd * scale

    # Consistency
    streak_bonus_per_trade: float = 0.02  # bonus per consecutive win
    max_streak_bonus: float = 0.2        # cap on streak bonus

    # Activity
    hold_penalty_per_bar: float = 0.001  # small penalty per bar held
    idle_penalty: float = 0.005          # penalty per bar while flat (encourages trading)

    # Opportunity cost
    missed_signal_penalty: float = 0.01  # penalty for ignoring strong signal while flat


def compute_reward(
    *,
    trade_pnl: float | None = None,
    starting_equity: float = 2000.0,
    current_equity: float = 2000.0,
    peak_equity: float = 2000.0,
    recent_pnls: list[float] | None = None,
    consecutive_wins: int = 0,
    bars_held: int = 0,
    is_flat: bool = True,
    missed_signal: bool = False,
    cfg: RewardConfig | None = None,
) -> float:
    """Compute shaped reward for one environment step.

    Args:
        trade_pnl: Realized P&L if a trade closed this step, else None
        starting_equity: Starting equity for the episode
        current_equity: Current equity
        peak_equity: Peak equity reached
        recent_pnls: List of recent trade P&Ls (for rolling Sharpe)
        consecutive_wins: Number of consecutive winning trades
        bars_held: Bars the current position has been held
        is_flat: Whether we have no open position
        missed_signal: Whether a strong signal was ignored while flat
        cfg: Reward configuration

    Returns:
        float: reward signal
    """
    c = cfg or RewardConfig()
    reward = 0.0

    # 1. P&L component (when a trade closes)
    if trade_pnl is not None:
        pnl_frac = trade_pnl / max(starting_equity, 1.0)
        reward += pnl_frac * c.pnl_scale
        if trade_pnl > 0:
            reward += c.win_bonus
        else:
            reward -= c.loss_penalty

    # 2. Risk-adjusted component (rolling Sharpe)
    if recent_pnls and len(recent_pnls) >= 3:
        window = recent_pnls[-c.sharpe_window:]
        returns = np.array(window) / max(starting_equity, 1.0)
        if np.std(returns) > 1e-9:
            rolling_sharpe = np.mean(returns) / np.std(returns)
            reward += rolling_sharpe * c.sharpe_bonus_scale

    # 3. Drawdown penalty
    drawdown = (peak_equity - current_equity) / max(peak_equity, 1.0)
    if drawdown > c.dd_threshold:
        reward -= (drawdown - c.dd_threshold) * c.dd_penalty_scale

    # 4. Consistency bonus (consecutive wins)
    if consecutive_wins > 1:
        streak_bonus = min(consecutive_wins * c.streak_bonus_per_trade, c.max_streak_bonus)
        reward += streak_bonus

    # 5. Activity penalties
    if not is_flat and bars_held > 0:
        reward -= c.hold_penalty_per_bar * bars_held

    if is_flat:
        reward -= c.idle_penalty

    # 6. Opportunity cost
    if missed_signal and is_flat:
        reward -= c.missed_signal_penalty

    return float(reward)
