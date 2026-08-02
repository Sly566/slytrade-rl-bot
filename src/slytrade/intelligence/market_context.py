"""Market Context Engine – understands current market state"""
from typing import Literal

import pandas as pd

from slytrade.config.trader_personality import TraderPersonality

MarketRegime = Literal["trending", "ranging", "high_volatility", "low_volatility", "transition"]

class MarketContextEngine:
    def __init__(self, personality: TraderPersonality):
        self.personality = personality

    def analyze(self, bars: pd.DataFrame, higher_tf_data: dict = None) -> dict:
        context = {}

        # Volatility detection
        if "atr_norm" in bars.columns:
            avg_atr = bars["atr_norm"].tail(20).mean()
            context["volatility_state"] = "high" if avg_atr > 0.8 else "normal"

        # Session detection (basic)
        context["session"] = "unknown"

        # Macro bias strength
        if higher_tf_data:
            context["macro_strength"] = "strong" if len(higher_tf_data) > 2 else "moderate"

        return context
