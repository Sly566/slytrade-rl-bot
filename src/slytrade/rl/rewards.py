"""Risk-adjusted reward shaping for the RL trading environment.

The default environment reward is raw equity delta, which trains agents that
chase variance and blow up drawdowns. Production trading rewards *consistent,
smooth* profitability. This module provides a composable reward function that
penalises drawdown and excessive turnover while still rewarding positive,
risk-adjusted returns. It is optional and opt-in so existing experiments are
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    # Weights for the composite reward. Set any weight to 0.0 to disable it.
    return_weight: float = 1.0
    drawdown_penalty_weight: float = 0.5
    turnover_penalty_weight: float = 0.05
    sharpe_weight: float = 0.0  # requires a return window to be meaningful
    # Drawdown penalty grows quadratically beyond this fraction of equity.
    drawdown_tolerance: float = 0.05
    # Transaction cost applied per unit turnover (already in price terms).
    transaction_cost: float = 0.0002
    clip: float = 10.0


def _clip(value: float, limit: float) -> float:
    if limit <= 0:
        return value
    return max(-limit, min(limit, value))


def shaped_reward(
    *,
    previous_equity: float,
    equity: float,
    position: float,
    target_position: float,
    peak_equity: float,
    config: RewardConfig | None = None,
) -> float:
    """Composite risk-adjusted reward for a single step.

    Parameters
    ----------
    previous_equity, equity:
        Equity before/after the step.
    position, target_position:
        Exposure before/after the step (turnover = abs(target - position)).
    peak_equity:
        Running peak equity used to measure drawdown.
    config:
        Weights/tolerances (defaults are conservative).
    """
    cfg = config or RewardConfig()
    equity_delta = equity - previous_equity
    base = max(previous_equity, 1e-9)
    ret = equity_delta / base

    turnover = abs(target_position - position)

    # Drawdown penalty: smooth near tolerance, quadratic beyond it.
    drawdown = (peak_equity - equity) / max(peak_equity, 1e-9)
    if drawdown <= cfg.drawdown_tolerance:
        dd_penalty = 0.0
    else:
        excess = drawdown - cfg.drawdown_tolerance
        dd_penalty = excess * excess

    cost_penalty = turnover * cfg.transaction_cost

    reward = (
        cfg.return_weight * ret
        - cfg.drawdown_penalty_weight * dd_penalty
        - cfg.turnover_penalty_weight * turnover
        - cost_penalty
    )
    return _clip(reward, cfg.clip)


def sharpe_of_returns(returns: list[float], *, floor: float = 1e-12) -> float:
    """Non-annualised Sharpe of a sequence of per-step returns (for metrics)."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = variance**0.5
    if std <= floor:
        return 0.0
    return mean / std
