"""Market regime engine.

Detects actionable market regimes from causal feature columns. The regime is
the single source of truth that drives personality adaptation, strategy
thresholds and position sizing. Everything here is causal: each bar only uses
data available up to that bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from slytrade.features.sessions import session_label

VolatilityRegime = Literal["low", "normal", "high"]
TrendRegime = Literal["bull", "bear", "ranging"]
SessionLabel = Literal["asia", "london", "ny_am", "ny_pm", "other", "unknown"]

REGIME_WEIGHT: dict[VolatilityRegime, float] = {"low": 0.7, "normal": 1.0, "high": 0.85}


@dataclass(frozen=True)
class MarketRegime:
    """Snapshot of the current market state."""

    volatility: VolatilityRegime
    trend: TrendRegime
    session: str
    volatility_zscore: float
    trend_strength_raw: float
    premium_discount: float
    score: float = 1.0  # 0..1 tradeability quality score

    @property
    def trending(self) -> bool:
        return self.trend in {"bull", "bear"}


class MarketRegimeEngine:
    """Adaptive market regime detector.

    The engine is intentionally simple and deterministic. It classifies the
    market into a volatility regime (rolling z-score of ATR-normalized price
    movement) and a trend regime (EMA spread sign) which the strategy and RL
    layers consume.
    """

    def __init__(
        self,
        *,
        volatility_lookback: int = 100,
        volatile_z_threshold: float = 0.8,
        quiet_z_threshold: float = -0.8,
        trend_threshold_atr: float = 0.15,
    ) -> None:
        if volatility_lookback < 20:
            raise ValueError("volatility_lookback must be >= 20")
        if volatile_z_threshold <= quiet_z_threshold:
            raise ValueError("volatile_z_threshold must exceed quiet_z_threshold")
        if trend_threshold_atr < 0:
            raise ValueError("trend_threshold_atr cannot be negative")
        self.volatility_lookback = volatility_lookback
        self.volatile_z_threshold = volatile_z_threshold
        self.quiet_z_threshold = quiet_z_threshold
        self.trend_threshold_atr = trend_threshold_atr

    def analyze_frame(self, bars: pd.DataFrame) -> pd.Series:
        """Return a per-bar MarketRegime snapshot series (index aligned)."""
        if bars.empty:
            return pd.Series(dtype=object, index=bars.index)
        atr_norm = _column_or_zeros(bars, "atr_norm")
        trend_strength = _column_or_zeros(bars, "trend_strength")
        premium_discount = _column_or_zeros(bars, "premium_discount")

        z = (atr_norm - atr_norm.rolling(self.volatility_lookback, min_periods=20).mean()) / atr_norm.rolling(
            self.volatility_lookback, min_periods=20
        ).std().replace(0.0, np.nan)

        regimes: list[MarketRegime] = []
        for i in range(len(bars)):
            z_i = float(z.iloc[i]) if pd.notna(z.iloc[i]) else 0.0
            volatility: VolatilityRegime = (
                "high" if z_i > self.volatile_z_threshold else ("low" if z_i < self.quiet_z_threshold else "normal")
            )
            ts = float(trend_strength.iloc[i])
            trend: TrendRegime = (
                "bull"
                if ts > self.trend_threshold_atr
                else ("bear" if ts < -self.trend_threshold_atr else "ranging")
            )
            session = _row_session(bars, i)
            premium = float(premium_discount.iloc[i])

            regimes.append(
                MarketRegime(
                    volatility=volatility,
                    trend=trend,
                    session=session,
                    volatility_zscore=z_i,
                    trend_strength_raw=ts,
                    premium_discount=premium,
                    score=0.0,  # overwritten below
                )
            )
        frame = pd.DataFrame([vars(r) for r in regimes], index=bars.index)
        frame["score"] = frame.apply(
            lambda r: _quality_score(r["volatility"], r["trend"], r["session"]), axis=1
        )
        return frame.apply(
            lambda r: MarketRegime(
                volatility=r["volatility"],
                trend=r["trend"],
                session=r["session"],
                volatility_zscore=float(r["volatility_zscore"]),
                trend_strength_raw=float(r["trend_strength_raw"]),
                premium_discount=float(r["premium_discount"]),
                score=float(r["score"]),
            ),
            axis=1,
        )

    def analyze_tail(self, bars: pd.DataFrame, tail: int = 1) -> MarketRegime:
        """Analyze bars and return the most recent regime snapshot."""
        if bars.empty:
            return MarketRegime("normal", "ranging", "unknown", 0.0, 0.0, 0.0, score=0.0)
        regimes = self.analyze_frame(bars)
        return regimes.iloc[-1]


def _column_or_zeros(bars: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric feature column, defaulting safely for sparse bars."""
    if name not in bars.columns:
        return pd.Series(0.0, index=bars.index, dtype=float)
    return pd.to_numeric(bars[name], errors="coerce").fillna(0.0)


def _row_session(bars: pd.DataFrame, i: int) -> str:
    """Resolve the trading session for bar row `i`.

    Prefers the bar's `time` column (canonical). Falls back to the DataFrame
    index when it is datetime-like, and finally to "unknown" so that session
    gates never crash on synthetic data.
    """
    if "time" in bars.columns:
        value = bars.iloc[i]["time"]
        if pd.notna(value):
            try:
                return session_label(pd.Timestamp(value).to_pydatetime())
            except (ValueError, TypeError):
                pass
    index_value = bars.index[i]
    if isinstance(index_value, (pd.Timestamp, np.datetime64)):
        return session_label(pd.Timestamp(index_value).to_pydatetime())
    return "unknown"


def _quality_score(volatility: VolatilityRegime, trend: TrendRegime, session: str) -> float:
    """Score tradeability from 0..1. Trend + normal volatility = best.

    Session windows with the most institutional flow score higher. This is a
    heuristic filter, not a profit prediction; it only ranks *when* to trade.
    """
    vol_score = REGIME_WEIGHT[volatility]
    trend_score = 0.75 if trend != "ranging" else 0.45

    session_score: float
    if session in {"london", "ny_am", "ny_pm"}:
        session_score = 1.0
    elif session == "asia":
        session_score = 0.6
    else:
        session_score = 0.4

    return min(0.35 * vol_score + 0.35 * trend_score + 0.30 * session_score, 1.0)
