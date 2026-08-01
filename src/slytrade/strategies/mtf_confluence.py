"""MTF ICT Confluence Strategy - requires macro + micro alignment."""
import pandas as pd

from slytrade.strategies.baselines import ICTConfluenceStrategy


class MTFICTConfluenceStrategy(ICTConfluenceStrategy):
    name = "mtf-ict-confluence"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_mtf_score = kwargs.get("min_mtf_score", 2)
        self.require_mtf_bias_alignment = kwargs.get("require_mtf_bias_alignment", True)

    def generate_signals(self, bars: pd.DataFrame, **kwargs):
        base_signals = super().generate_signals(bars, **kwargs)  # type: ignore[misc]
        if "mtf_confluence_score" not in bars.columns:
            return base_signals
        mtf_ok = bars["mtf_confluence_score"] >= self.min_mtf_score
        if self.require_mtf_bias_alignment and "mtf_bias" in bars.columns:
            bias = bars["mtf_bias"]
            direction = base_signals.astype(int).replace({True: 1, False: -1})
            alignment = (bias == direction) | (bias == 0)
            return base_signals & mtf_ok & alignment
        return base_signals & mtf_ok
