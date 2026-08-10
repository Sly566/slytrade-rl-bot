"""Personality-adaptive ICT strategy.

The engine of the "ICT trader persona": the same causal ICT confluence scores
from the baseline strategy, but entry thresholds, direction filters and
position sizing are modulated in real time by the trader's personality traits
and the detected market regime.

Everything here is causal: decisions at bar *i* only use data from bars
`<= i`. The strategy keeps an internal rolling window so it can run inside any
of the existing backtest engines that call `on_bar(index, bar)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.execution.models import OrderIntent, Side
from slytrade.intelligence.market_context import MarketContextEngine
from slytrade.intelligence.micro_macro_alignment import MicroMacroAlignmentEngine
from slytrade.intelligence.regime import MarketRegimeEngine
from slytrade.strategies.baselines import ICTConfluenceStrategy

ADAPTIVE_RULE_KEYS = {
    "high": ["high_volatility", "high_volatility_selectivity"],
    "low": ["low_volatility"],
    "london": ["london_open"],
    "ny": ["ny_open"],
}


@dataclass(frozen=True)
class PersonalityAdaptiveConfig:
    """Configuration for the personality-adaptive strategy."""

    min_score: int = 4
    cooldown_bars: int = 20
    allowed_sessions: tuple[str, ...] = ("london", "ny_am", "ny_pm")
    require_fresh_quote: bool = True
    max_spread: float | None = None
    min_tick_rate: float = 0.0
    min_abs_trend_strength: float = 0.0
    history_window: int = 120

    # Risk-based position sizing
    risk_based_sizing: bool = True
    equity: float = 100_000.0
    risk_per_trade: float = 0.005
    point_value: float = 1.0
    stop_loss_atr: float = 1.0
    min_slot_distance_price: float = 0.10

    # Regime / alignment gates
    use_regime_filter: bool = True
    require_mtf_alignment: bool = True
    alignment_threshold: float = 0.6
    regime_filter_penalty: float = field(default=1.5)  # extra score required when regime is poor


class PersonalityAdaptiveStrategy:
    """ICT confluence strategy whose behavior adapts to the market regime.

    Thresholds shift with volatility, trend and session; direction is gated by
    the trader's regime preferences; position size is risk-budgeted instead of
    fixed. The strategy never executes orders - it only emits OrderIntents.
    """

    name = "personality-adaptive"

    def __init__(
        self,
        personality: TraderPersonality | None = None,
        config: PersonalityAdaptiveConfig | None = None,
        *,
        symbol: str = "XAUUSD",
        volume: float = 0.1,
    ) -> None:
        if volume <= 0:
            raise ValueError("volume must be positive")
        self.symbol = symbol
        self.volume = volume
        self.personality = personality or TraderPersonality.from_yaml()
        self.config = config or PersonalityAdaptiveConfig()

        self._scorer = ICTConfluenceStrategy(
            symbol=symbol,
            volume=volume,
            min_score=1,
            cooldown_bars=0,
            discount_threshold=-0.15,
            premium_threshold=0.15,
            min_abs_trend_strength=self.config.min_abs_trend_strength,
            allowed_sessions=self.config.allowed_sessions,
            require_fresh_quote=self.config.require_fresh_quote,
            max_spread=self.config.max_spread,
            min_tick_rate=self.config.min_tick_rate,
        )
        self._context_engine = MarketContextEngine(self.personality, MarketRegimeEngine())
        self._alignment_engine = MicroMacroAlignmentEngine(self.personality)

        self._history: list[pd.Series] = []
        self._side: str = "flat"
        self._last_entry_index: int = -1_000_000
        self._trades: list[OrderIntent] = []

    # ------------------------------------------------------------------
    # Public API (used by backtest engines)
    # ------------------------------------------------------------------
    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        """Return an OrderIntent when the persona wants to enter, else None."""
        self._history.append(bar.copy())
        if len(self._history) > self.config.history_window:
            self._history.pop(0)

        if index - self._last_entry_index < self.config.cooldown_bars:
            return None
        if self.config.require_fresh_quote and not bool(bar.get("quote_is_fresh", True)):
            return None
        if not _session_allowed(bar, self.config.allowed_sessions):
            return None

        context = self._context_engine.analyze(pd.DataFrame(self._history))
        alignment = self._alignment_engine.evaluate(pd.DataFrame(self._history), context)

        # Regime gates -------------------------------------------------
        if self.config.use_regime_filter:
            if context["volatility"] not in self.personality.volatility_preferences:
                return None
            if context["trend"] not in self.personality.trend_preferences:
                return None
            if context["session"] not in self.personality.session_preferences:
                return None

        if self.config.require_mtf_alignment and alignment < self.config.alignment_threshold:
            return None

        # Personality-adaptive threshold -------------------------------
        threshold = self._adaptive_threshold(context)
        long_score = self._scorer._long_score(bar)  # type: ignore[attr-defined]
        short_score = self._scorer._short_score(bar)  # type: ignore[attr-defined]

        # Poor regime: demand a stronger setup (adaptive discipline)
        if float(context.get("regime_score", 0.5)) < 0.5:
            threshold = int(round(threshold + self.config.regime_filter_penalty))

        intent: OrderIntent | None = None
        if long_score >= threshold and long_score > short_score and self._side != "long":
            intent = self._build_intent(Side.BUY, long_score, bar)
        elif short_score >= threshold and short_score > long_score and self._side != "short":
            intent = self._build_intent(Side.SELL, short_score, bar)

        if intent is not None:
            self._side = "long" if intent.side == Side.BUY else "short"
            self._last_entry_index = index
            self._trades.append(intent)
        return intent

    def reset(self) -> None:
        """Reset in-strategy state (used between episodes/walk-forward folds)."""
        self._history.clear()
        self._side = "flat"
        self._last_entry_index = -1_000_000
        self._trades.clear()

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _adaptive_threshold(self, context: dict) -> int:
        """Compute the entry-score threshold from personality + regime.

        Higher selectivity/conviction & worse regimes -> higher threshold.
        Higher aggression/adaptability & favorable regimes -> lower threshold.
        """
        personality = self.personality
        base = float(personality.confidence_thresholds.get("min_entry_score", self.config.min_score))

        # Selectivity raises the bar; aggression lowers it.
        threshold = base + (personality.selectivity - 0.5) * 2.0 - (personality.aggression - 0.5) * 2.0

        # Edge optimism: skeptical personas need stronger confirmations.
        threshold += (0.5 - personality.edge_optimism) * 2.0

        # Regime shifts
        volatility = context.get("volatility", "normal")
        if volatility == "high":
            threshold += 1.0 * personality.adaptability  # high vol -> be picky
        elif volatility == "low":
            threshold -= 1.0 * personality.adaptability  # low vol -> easier edges
        if context.get("macro_strength") == "weak":
            threshold += 1.0 * personality.structure_focus

        # Session boost
        if context.get("session") in personality.session_preferences:
            threshold -= 0.5 * personality.session_sensitivity

        return max(1, int(round(threshold)))

    def _build_intent(self, side: Side, score: int, bar: pd.Series) -> OrderIntent:
        """Create the OrderIntent, applying risk-based sizing when enabled."""
        reason = f"persona_{side.value}_{score}"
        if self.config.risk_based_sizing:
            volume = self._risk_sized_volume(bar)
        else:
            volume = self.volume
        return OrderIntent(symbol=self.symbol, side=side, volume=volume, reason=reason)

    def _risk_sized_volume(self, bar: pd.Series) -> float:
        """Size so a stop-out loses approximately risk_per_trade of equity."""
        atr = float(bar.get("atr", 0.0) or 0.0)
        stop_distance = max(atr * self.config.stop_loss_atr, self.config.min_slot_distance_price)
        risk_budget = self.config.equity * self.config.risk_per_trade
        denominator = stop_distance * self.config.point_value
        if denominator <= 0:
            return self.volume
        volume = risk_budget / denominator
        if volume <= 0 or not volume == volume:  # NaN guard
            return self.volume
        return round(float(volume), 8)


def _session_allowed(bar: pd.Series, allowed_sessions: tuple[str, ...]) -> bool:
    if not allowed_sessions:
        return True
    session_columns = {
        "asia": "session_asia",
        "london": "session_london",
        "ny_am": "session_ny_am",
        "ny_pm": "session_ny_pm",
        "other": "session_other",
    }
    present = [column for column in session_columns.values() if column in bar.index]
    if not present:
        return True
    for session in allowed_sessions:
        column = session_columns.get(session)
        if column and float(bar.get(column, 0.0)) > 0:
            return True
    return False
