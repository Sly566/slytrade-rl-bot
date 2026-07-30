"""Baseline and future strategy implementations."""

from slytrade.strategies.baselines import (
    BuyAndHoldStrategy,
    ICTBiasBaselineStrategy,
    MovingAverageCrossStrategy,
    NoTradeStrategy,
)

__all__ = [
    "NoTradeStrategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy",
    "ICTBiasBaselineStrategy",
]
