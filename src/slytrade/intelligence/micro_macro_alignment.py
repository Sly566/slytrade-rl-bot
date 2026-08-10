"""Micro-Macro Alignment Engine.

Evaluates how well the execution-timeframe setup aligns with the higher
timeframe (macro) structure. A high score means the micro signal is trading
with the macro flow instead of against it.
"""
from __future__ import annotations

import pandas as pd

from slytrade.config.trader_personality import TraderPersonality


class MicroMacroAlignmentEngine:
    def __init__(self, personality: TraderPersonality):
        self.personality = personality

    def evaluate(self, bars: pd.DataFrame, context: dict) -> float:
        """Return an alignment score in [0, 1].

        Combines:
        - macro structure strength (from context['macro_strength'])
        - MTF bias alignment (if the execution frame carries 'mtf_bias')
        - regime quality (context['regime_score']) weighted by macro respect
        """
        score = 0.5

        macro_strength = context.get("macro_strength", "moderate")
        if macro_strength == "strong":
            score += 0.15 * self.personality.macro_respect
        elif macro_strength == "weak":
            score -= 0.10 * self.personality.macro_respect

        # MTF bias alignment: micro trend vs macro bias
        mtf_bias = context.get("mtf_bias")
        trend_raw = context.get("trend_strength_raw", 0.0)
        if mtf_bias is not None:
            if (mtf_bias > 0 and trend_raw > 0) or (mtf_bias < 0 and trend_raw < 0):
                score += 0.20 * self.personality.structure_focus
            elif mtf_bias != 0 and (mtf_bias > 0) != (trend_raw > 0):
                score -= 0.20 * self.personality.structure_focus

        # Regime quality: only trade quality windows
        regime_score = float(context.get("regime_score", 0.5))
        score += (regime_score - 0.5) * 0.2 * self.personality.macro_respect

        # Session preference
        session = context.get("session", "unknown")
        if session in self.personality.session_preferences:
            score += 0.1 * self.personality.session_sensitivity

        return float(min(max(score, 0.0), 1.0))
