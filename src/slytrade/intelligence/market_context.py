"""Market Context Engine"""

import pandas as pd

from slytrade.config.trader_personality import TraderPersonality


class MarketContextEngine:
    def __init__(self, personality: TraderPersonality):
        self.personality = personality

    def analyze(self, bars: pd.DataFrame, higher_tf_data: dict | None = None) -> dict:
        context = {}

        # Volatility
        if "atr_norm" in bars.columns:
            avg_atr = bars["atr_norm"].tail(20).mean()
            context["volatility"] = "high" if avg_atr > 0.8 else "normal"

        # Macro strength
        if higher_tf_data:
            context["macro_strength"] = "strong" if len(higher_tf_data) >= 3 else "moderate"

        # Session placeholder (expandable)
        context["session"] = "unknown"

        return context
