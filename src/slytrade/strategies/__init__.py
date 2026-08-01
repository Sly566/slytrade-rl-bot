"""Baseline and future strategy implementations."""

from slytrade.strategies.baselines import (
    BuyAndHoldStrategy,
    ICTBiasBaselineStrategy,
    ICTConfluenceStrategy,
    MovingAverageCrossStrategy,
    NoTradeStrategy,
)

__all__ = [
    "NoTradeStrategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy",
    "ICTBiasBaselineStrategy",
    "ICTConfluenceStrategy",
]
