"""Position sizing for risk-budgeted and Kelly-style allocation.

Every strategy and the paper loop should size positions from the same functions
so that risk-per-trade is applied consistently across backtests, paper trading
and (eventually) live trading. Volume is always normalised to the broker's
min/max/step constraints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SizingParams:
    risk_per_trade: float = 0.005
    point_value: float = 1.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    kelly_fraction: float = 0.25  # fraction of the full Kelly criterion to stake


def normalize_volume(
    volume: float,
    *,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
    volume_step: float = 0.01,
) -> float:
    """Clamp and round ``volume`` onto the broker's volume ladder."""
    if volume <= 0:
        return 0.0
    step = volume_step if volume_step > 0 else 0.01
    minimum = max(volume_min, 0.0)
    clamped = min(max(volume, minimum), volume_max)
    steps = round((clamped - minimum) / step)
    return round(minimum + steps * step, 10)


def risk_based_volume(
    equity: float,
    stop_distance: float,
    *,
    risk_per_trade: float = 0.005,
    point_value: float = 1.0,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
    volume_step: float = 0.01,
    floor_distance: float = 1e-9,
) -> float:
    """Size a position so a stop-out loses ~``risk_per_trade`` of equity.

    ``stop_distance`` is expressed in price units; ``point_value`` is the PnL
    per price unit per 1.0 volume. A non-positive stop distance yields zero
    volume (fail safe) rather than an absurdly large position.
    """
    if equity <= 0 or stop_distance <= floor_distance or risk_per_trade <= 0:
        return 0.0
    risk_budget = equity * risk_per_trade
    raw = risk_budget / (stop_distance * point_value)
    return normalize_volume(raw, volume_min=volume_min, volume_max=volume_max, volume_step=volume_step)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Return the full-Kelly fraction (f* = p - q/b), clamped to [0, 1).

    ``avg_loss`` must be positive (magnitude). Degenerate inputs yield 0.0 so a
    bad statistic can never produce a negative or oversized allocation.
    """
    if avg_loss <= 0 or not (0.0 < win_rate < 1.0):
        return 0.0
    b = avg_win / avg_loss
    if b <= 0:
        return 0.0
    q = 1.0 - win_rate
    fraction = win_rate - q / b
    return max(0.0, min(0.5, fraction))


def kelly_volume(
    equity: float,
    stop_distance: float,
    *,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    point_value: float = 1.0,
    kelly_fraction_of: float = 0.25,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
    volume_step: float = 0.01,
) -> float:
    """Size using a fractional-Kelly criterion with a stop-distance budget.

    The stake (in currency) is ``equity * f * kelly_fraction_of``; the volume is
    derived from the stop distance the same way as ``risk_based_volume``.
    """
    fraction = kelly_fraction(win_rate, avg_win, avg_loss)
    if fraction <= 0 or stop_distance <= 0 or point_value <= 0:
        return 0.0
    stake = equity * fraction * kelly_fraction_of
    return normalize_volume(
        stake / (stop_distance * point_value),
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
    )
