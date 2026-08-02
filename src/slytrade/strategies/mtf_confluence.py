"""MTF ICT Confluence Strategy with Trader Personality"""

import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.strategies.baselines import ICTConfluenceStrategy


class MTFICTConfluenceStrategy(ICTConfluenceStrategy):
    name = "mtf-ict-confluence"

    def __init__(self, personality: TraderPersonality | None = None, **kwargs):
        super().__init__(**kwargs)
        self.personality = personality or TraderPersonality.from_yaml()

        # Dynamic threshold based on personality
        self.min_mtf_score = int(2 * self.personality.selectivity)
        self.require_mtf_bias_alignment = True

    def generate_signals(self, bars: pd.DataFrame, **kwargs):
        base_signals = super().generate_signals(bars, **kwargs)  # type: ignore[misc]

        if "mtf_confluence_score" not in bars.columns:
            return base_signals

        mtf_ok = bars["mtf_confluence_score"] >= self.min_mtf_score
        return base_signals & mtf_ok
