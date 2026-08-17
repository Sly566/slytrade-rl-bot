"""Market Context Engine.

Provides a rich, causal market-context snapshot that the strategy layer uses to
adapt its behavior. The context is derived from the market regime engine, MTF
data and macro-alignment input, and is deliberately deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.intelligence.regime import MarketRegime, MarketRegimeEngine

DEFAULT_CONTEXT: dict = {
    "volatility": "normal",
    "trend": "ranging",
    "session": "unknown",
    "regime_score": 0.5,
    "macro_strength": "moderate",
}


class MarketContextEngine:
    """Build a market-context snapshot from bars and higher-timeframe data.

    The returned dict keeps backward compatibility with the original interface
    (volatility, macro_strength, session) while adding regime and MTF fields.
    """

    def __init__(
        self,
        personality: TraderPersonality,
        regime_engine: MarketRegimeEngine | None = None,
    ):
        self.personality = personality
        self.regime_engine = regime_engine or MarketRegimeEngine()

    def analyze(self, bars: pd.DataFrame, higher_tf_data: dict | None = None) -> dict:
        """Analyze bars (plus optional HTF frames) and return a context dict."""
        if bars is None or bars.empty:
            return dict(DEFAULT_CONTEXT)

        # Regime from the execution timeframe
        regime: MarketRegime = self.regime_engine.analyze_tail(bars)

        context = {
            "volatility": regime.volatility,
            "trend": regime.trend,
            "session": regime.session,
            "regime_score": regime.score,
            "volatility_zscore": regime.volatility_zscore,
            "trend_strength_raw": regime.trend_strength_raw,
            "premium_discount": regime.premium_discount,
        }

        # Aggregate volatility from charted ATR if present
        if "atr_norm" in bars.columns:
            avg_atr = float(pd.to_numeric(bars["atr_norm"], errors="coerce").fillna(0.0).tail(20).mean())
            context["atr_norm_20"] = avg_atr

        # MTF macro strength
        macro_strength = "moderate"
        if higher_tf_data:
            macro_score = 0
            for tf, frame in higher_tf_data.items():
                if frame is None or frame.empty:
                    continue
                bias_col = f"htf_{tf.lower()}_bos_dir"
                if bias_col in frame.columns:
                    last = pd.to_numeric(frame[bias_col], errors="coerce").fillna(0.0).iloc[-1]
                    if last != 0:
                        macro_score += 1
                elif "bos_dir" in frame.columns:
                    last = pd.to_numeric(frame["bos_dir"], errors="coerce").fillna(0.0).iloc[-1]
                    if last != 0:
                        macro_score += 1
            if macro_score >= 3:
                macro_strength = "strong"
            elif macro_score == 0:
                macro_strength = "weak"
        context["macro_strength"] = macro_strength
        # Higher-timeframe data is available either when frames are passed
        # explicitly, or when the aligned bars already carry MTF features
        # (mtf_bias / mtf_confluence_score / htf_* columns from the pipeline's
        # MTF injection step). Without this, downstream alignment gates must
        # treat "no evidence" as neutral, NOT as a weak/misaligned structure.
        context["has_htf"] = bool(higher_tf_data) or any(
            column in bars.columns for column in ("mtf_bias", "mtf_confluence_score")
        )

        # Aggregate MTF bias alignment if the execution frame already carries it
        if "mtf_bias" in bars.columns:
            mtf_bias = pd.to_numeric(bars["mtf_bias"], errors="coerce").fillna(0.0).iloc[-1]
            context["mtf_bias"] = int(mtf_bias)
        if "mtf_confluence_score" in bars.columns:
            mtf_score = int(pd.to_numeric(bars["mtf_confluence_score"], errors="coerce").fillna(0).iloc[-1])
            context["mtf_confluence_score"] = mtf_score

        return context

    def analyze_tail_arrays(
        self,
        *,
        atr_norm: np.ndarray,
        trend_strength: np.ndarray,
        premium_discount: np.ndarray,
        times: list,
        mtf_bias: np.ndarray | None = None,
        mtf_confluence: np.ndarray | None = None,
        has_htf: bool = False,
    ) -> dict:
        """Tail-only context from pre-buffered scalar windows.

        Equivalent to :meth:`analyze` for the trailing bar, but O(1) in the
        window length — the persona strategy calls this every bar, so it must
        never rebuild a DataFrame or re-classify the whole window.
        """
        regime = self.regime_engine.analyze_tail_arrays(atr_norm, trend_strength, premium_discount, times)
        context = {
            "volatility": regime.volatility,
            "trend": regime.trend,
            "session": regime.session,
            "regime_score": regime.score,
            "volatility_zscore": regime.volatility_zscore,
            "trend_strength_raw": regime.trend_strength_raw,
            "premium_discount": regime.premium_discount,
        }
        if atr_norm.size:
            context["atr_norm_20"] = float(np.mean(atr_norm[-20:]))
        context["macro_strength"] = "moderate"
        context["has_htf"] = has_htf
        if mtf_bias is not None and mtf_bias.size:
            context["mtf_bias"] = int(mtf_bias[-1])
        if mtf_confluence is not None and mtf_confluence.size:
            context["mtf_confluence_score"] = int(mtf_confluence[-1])
        return context
