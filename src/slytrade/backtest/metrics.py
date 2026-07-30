from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerformanceMetrics:
    start_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe_like: float
    equity_points: int
    trades: int


def compute_max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    equity = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / np.maximum(peaks, 1e-12)
    return float(np.max(drawdowns))


def compute_sharpe_like(equity_curve: list[float]) -> float:
    """Return a simple non-annualized Sharpe-like score for early backtests."""
    if len(equity_curve) < 3:
        return 0.0
    equity = np.asarray(equity_curve, dtype=float)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    std = float(np.std(returns))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / std)


def compute_performance_metrics(equity_curve: list[float], trades: int = 0) -> PerformanceMetrics:
    if not equity_curve:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, trades)
    start = float(equity_curve[0])
    final = float(equity_curve[-1])
    total_return = (final - start) / max(start, 1e-12)
    return PerformanceMetrics(
        start_equity=start,
        final_equity=final,
        total_return=float(total_return),
        max_drawdown=compute_max_drawdown(equity_curve),
        sharpe_like=compute_sharpe_like(equity_curve),
        equity_points=len(equity_curve),
        trades=trades,
    )
