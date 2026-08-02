"""MTF ICT Confluence Strategy with Trader Personality"""
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.intelligence.market_context import MarketContextEngine
from slytrade.intelligence.micro_macro_alignment import MicroMacroAlignmentEngine
from slytrade.strategies.baselines import ICTConfluenceStrategy


class MTFICTConfluenceStrategy(ICTConfluenceStrategy):
    name = "mtf-ict-confluence"

    def __init__(self, personality: TraderPersonality = None, **kwargs):
        super().__init__(**kwargs)
        self.personality = personality or TraderPersonality.from_yaml()
        self.context_engine = MarketContextEngine(self.personality)
        self.alignment_engine = MicroMacroAlignmentEngine(self.personality)

        # Dynamic threshold based on personality
        self.min_mtf_score = int(2 * self.personality.selectivity)
        self.require_mtf_bias_alignment = True

    def generate_signals(self, bars: pd.DataFrame, higher_tf_data: dict = None, **kwargs):
        context = self.context_engine.analyze(bars, higher_tf_data)
        alignment_score = self.alignment_engine.evaluate(bars, context)

        # Dynamic threshold based on personality and context
        dynamic_threshold = self.min_mtf_score
        if alignment_score > 0.75:
            dynamic_threshold = max(1, dynamic_threshold - 1)

        base_signals = super().generate_signals(bars, **kwargs)  # type: ignore[misc]

        if "mtf_confluence_score" not in bars.columns:
            return base_signals

        mtf_ok = bars["mtf_confluence_score"] >= dynamic_threshold
        return base_signals & mtf_ok
