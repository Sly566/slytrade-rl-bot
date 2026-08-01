"""MTF Strategy Tuning Helpers (Phase 24.1)."""
from slytrade.strategies.mtf_confluence import MTFICTConfluenceStrategy


def get_mtf_strategy(min_mtf_score=2, require_mtf_bias_alignment=True, **kwargs):
    return MTFICTConfluenceStrategy(
        min_mtf_score=min_mtf_score,
        require_mtf_bias_alignment=require_mtf_bias_alignment,
        **kwargs
    )
