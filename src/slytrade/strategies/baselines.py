from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from slytrade.execution.models import OrderIntent, Side

StrategySide = Literal["long", "short", "flat"]


def _close_from_bar(bar: pd.Series) -> float:
    return float(bar["close"])


@dataclass
class NoTradeStrategy:
    """Baseline that never trades.

    Every research report should include this baseline. A strategy that cannot
    beat no-trade after costs has no deployable edge.
    """

    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        return None


@dataclass
class BuyAndHoldStrategy:
    """Submit one buy order on the first eligible bar."""

    symbol: str
    volume: float
    reason: str = "buy_and_hold"
    _submitted: bool = field(default=False, init=False)

    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        if self._submitted:
            return None
        self._submitted = True
        return OrderIntent(symbol=self.symbol, side=Side.BUY, volume=self.volume, reason=self.reason)


@dataclass
class MovingAverageCrossStrategy:
    """Simple moving-average crossover baseline.

    This is not meant to be a final trading model. It is a sanity-check baseline
    that RL and ICT strategies must outperform after costs.
    """

    symbol: str
    volume: float
    fast_window: int = 5
    slow_window: int = 20
    allow_short: bool = True
    _closes: list[float] = field(default_factory=list, init=False)
    _previous_diff: float | None = field(default=None, init=False)
    _side: StrategySide = field(default="flat", init=False)

    def __post_init__(self) -> None:
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("moving-average windows must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")

    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        self._closes.append(_close_from_bar(bar))
        if len(self._closes) < self.slow_window:
            return None

        fast = sum(self._closes[-self.fast_window :]) / self.fast_window
        slow = sum(self._closes[-self.slow_window :]) / self.slow_window
        diff = fast - slow
        previous = self._previous_diff
        self._previous_diff = diff
        if previous is None:
            return None

        if previous <= 0 < diff and self._side != "long":
            self._side = "long"
            return OrderIntent(symbol=self.symbol, side=Side.BUY, volume=self.volume, reason="ma_cross_bullish")

        if previous >= 0 > diff and self.allow_short and self._side != "short":
            self._side = "short"
            return OrderIntent(symbol=self.symbol, side=Side.SELL, volume=self.volume, reason="ma_cross_bearish")

        return None


@dataclass
class ICTBiasBaselineStrategy:
    """Rule-based ICT/SMC baseline using causal feature columns.

    Long rule:
    - bullish BOS/CHOCH context
    - price in discount or neutral area

    Short rule:
    - bearish BOS/CHOCH context
    - price in premium or neutral area

    This deliberately stays simple. It exists to benchmark whether later RL
    policies genuinely add value beyond causal ICT features.
    """

    symbol: str
    volume: float
    discount_threshold: float = 0.0
    premium_threshold: float = 0.0
    use_choch: bool = True
    _side: StrategySide = field(default="flat", init=False)

    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        bos_dir = float(bar.get("bos_dir", 0.0))
        choch_dir = float(bar.get("choch_dir", 0.0)) if self.use_choch else 0.0
        premium_discount = float(bar.get("premium_discount", 0.0))
        liquidity_sweep = float(bar.get("liquidity_sweep", 0.0))

        bullish_context = bos_dir > 0 or choch_dir > 0 or liquidity_sweep < 0
        bearish_context = bos_dir < 0 or choch_dir < 0 or liquidity_sweep > 0

        if bullish_context and premium_discount <= self.discount_threshold and self._side != "long":
            self._side = "long"
            return OrderIntent(symbol=self.symbol, side=Side.BUY, volume=self.volume, reason="ict_bias_long")

        if bearish_context and premium_discount >= self.premium_threshold and self._side != "short":
            self._side = "short"
            return OrderIntent(symbol=self.symbol, side=Side.SELL, volume=self.volume, reason="ict_bias_short")

        return None
