from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from slytrade.execution.models import OrderIntent, Side

StrategySide = Literal["long", "short", "flat"]


def _close_from_bar(bar: pd.Series) -> float:
    return float(bar["close"])


def _is_session_allowed(bar: pd.Series, allowed_sessions: tuple[str, ...]) -> bool:
    if not allowed_sessions:
        return True
    session_columns = {
        "asia": "session_asia",
        "london": "session_london",
        "ny_am": "session_ny_am",
        "ny_pm": "session_ny_pm",
        "other": "session_other",
    }
    present_columns = [column for column in session_columns.values() if column in bar.index]
    if not present_columns:
        return True
    for session in allowed_sessions:
        column = session_columns.get(session)
        if column and float(bar.get(column, 0.0)) > 0:
            return True
    return False


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
    """Simple rule-based ICT/SMC baseline using causal feature columns."""

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


@dataclass
class ICTConfluenceStrategy:
    """Stricter ICT/SMC baseline designed for strategy tuning.

    This strategy does not try to be clever. It simply requires multiple causal
    ICT confirmations before allowing an entry and uses cooldown/session filters
    to reduce overtrading. It is the first serious hand-crafted benchmark that
    later RL policies should beat.
    """

    symbol: str
    volume: float
    min_score: int = 4
    cooldown_bars: int = 20
    discount_threshold: float = -0.15
    premium_threshold: float = 0.15
    min_tick_rate: float = 0.0
    max_spread: float | None = None
    min_abs_trend_strength: float = 0.0
    allowed_sessions: tuple[str, ...] = ("london", "ny_am", "ny_pm")
    require_fresh_quote: bool = True
    _side: StrategySide = field(default="flat", init=False)
    _last_entry_index: int = field(default=-10_000_000, init=False)

    def _can_trade(self, index: int, bar: pd.Series) -> bool:
        if index - self._last_entry_index < self.cooldown_bars:
            return False
        if self.require_fresh_quote and not bool(bar.get("quote_is_fresh", True)):
            return False
        if self.max_spread is not None and float(bar.get("quote_spread", 0.0)) > self.max_spread:
            return False
        if float(bar.get("tick_rate_per_second", 0.0)) < self.min_tick_rate:
            return False
        return _is_session_allowed(bar, self.allowed_sessions)

    def _long_score(self, bar: pd.Series) -> int:
        score = 0
        premium_discount = float(bar.get("premium_discount", 0.0))
        trend_strength = float(bar.get("trend_strength", 0.0))
        if float(bar.get("bos_dir", 0.0)) > 0:
            score += 2
        if float(bar.get("choch_dir", 0.0)) > 0:
            score += 1
        if float(bar.get("liquidity_sweep", 0.0)) < 0:
            score += 1
        if float(bar.get("fvg_bullish", 0.0)) > 0:
            score += 1
        if float(bar.get("order_block_bullish", 0.0)) > 0:
            score += 1
        if premium_discount <= self.discount_threshold:
            score += 1
        if trend_strength >= self.min_abs_trend_strength:
            score += 1
        return score

    def _short_score(self, bar: pd.Series) -> int:
        score = 0
        premium_discount = float(bar.get("premium_discount", 0.0))
        trend_strength = float(bar.get("trend_strength", 0.0))
        if float(bar.get("bos_dir", 0.0)) < 0:
            score += 2
        if float(bar.get("choch_dir", 0.0)) < 0:
            score += 1
        if float(bar.get("liquidity_sweep", 0.0)) > 0:
            score += 1
        if float(bar.get("fvg_bearish", 0.0)) > 0:
            score += 1
        if float(bar.get("order_block_bearish", 0.0)) > 0:
            score += 1
        if premium_discount >= self.premium_threshold:
            score += 1
        if trend_strength <= -self.min_abs_trend_strength:
            score += 1
        return score

    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        if not self._can_trade(index, bar):
            return None

        long_score = self._long_score(bar)
        short_score = self._short_score(bar)

        if long_score >= self.min_score and long_score > short_score and self._side != "long":
            self._side = "long"
            self._last_entry_index = index
            return OrderIntent(
                symbol=self.symbol,
                side=Side.BUY,
                volume=self.volume,
                reason=f"ict_confluence_long_{long_score}",
            )

        if short_score >= self.min_score and short_score > long_score and self._side != "short":
            self._side = "short"
            self._last_entry_index = index
            return OrderIntent(
                symbol=self.symbol,
                side=Side.SELL,
                volume=self.volume,
                reason=f"ict_confluence_short_{short_score}",
            )

        return None
