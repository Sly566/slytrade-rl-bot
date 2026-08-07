"""MTF ICT Confluence Strategy with Trader Personality"""
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.intelligence.market_context import MarketContextEngine
from slytrade.intelligence.micro_macro_alignment import MicroMacroAlignmentEngine
from slytrade.strategies.baselines import ICTConfluenceStrategy


class MTFICTConfluenceStrategy(ICTConfluenceStrategy):
    name = "mtf-ict-confluence"

    def __init__(self, personality: TraderPersonality | None = None, **kwargs):
        super().__init__(**kwargs)
        self.personality = personality or TraderPersonality.from_yaml()
        self.context_engine = MarketContextEngine(self.personality)
        self.alignment_engine = MicroMacroAlignmentEngine(self.personality)

        self.min_mtf_score = int(2 * self.personality.selectivity)
        self.require_mtf_bias_alignment = True

    def generate_signals(self, bars: pd.DataFrame, higher_tf_data: dict | None = None, **kwargs):
        context = self.context_engine.analyze(bars, higher_tf_data or {})
        alignment_score = self.alignment_engine.evaluate(bars, context)

        # === Dynamic Threshold Logic ===
        threshold = self.min_mtf_score

        # Aggression influence
        if self.personality.aggression > 0.8:
            threshold = max(1, threshold - 1)

        # Volatility influence
        if context.get("volatility") == "high":
            threshold += 1

        # Macro respect influence
        if context.get("macro_strength") == "strong" and self.personality.macro_respect > 0.7:
            threshold = max(1, threshold - 1)

        # Alignment bonus
        if alignment_score > 0.75:
            threshold = max(1, threshold - 1)

        # Session sensitivity (basic example)
        if context.get("session") == "london_open" and self.personality.session_sensitivity > 0.7:
            threshold = max(1, threshold - 1)

        base_signals = super().generate_signals(bars, **kwargs)  # type: ignore[misc]

        if "mtf_confluence_score" not in bars.columns:
            return base_signals

        mtf_ok = bars["mtf_confluence_score"] >= threshold
        return base_signals & mtf_ok
