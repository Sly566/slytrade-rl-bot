"""Micro-Macro Alignment Engine"""

import pandas as pd

from slytrade.config.trader_personality import TraderPersonality


class MicroMacroAlignmentEngine:
    def __init__(self, personality: TraderPersonality):
        self.personality = personality

    def evaluate(self, bars: pd.DataFrame, context: dict) -> float:
        score = 0.5
        if context.get("macro_strength") == "strong":
            score += 0.3 * self.personality.macro_respect
        return min(max(score, 0.0), 1.0)
