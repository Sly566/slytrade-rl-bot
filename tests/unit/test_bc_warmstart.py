"""Regression tests for the behavioural-cloning warmstart + persona action feature."""
from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.rl.dataset import persona_action_column
from slytrade.rl.walkforward import persona_actions_for_bars


def make_bars(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    bull = np.tile(np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0]), n // 6 + 1)[:n]
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
            "symbol": "XAUUSD", "timeframe": "M15",
            "open": close, "high": close + 0.4, "low": close - 0.4, "close": close,
            "tick_volume": 100.0, "spread": 2.0, "real_volume": 0.0,
            "atr": 0.4, "bos_dir": bull, "choch_dir": 0.0, "liquidity_sweep": -bull,
            "fvg_bullish": (bull > 0).astype(float), "fvg_bearish": (bull < 0).astype(float),
            "order_block_bullish": 0.0, "order_block_bearish": 0.0,
            "premium_discount": -0.5 * bull, "trend_strength": 0.5 * bull,
            "mtf_bias": bull, "mtf_confluence_score": 3.0,
            "htf_h4_trend_strength": np.full(n, 1.0),
        }
    )


def test_persona_action_column_is_causal_and_in_range():
    bars = make_bars(300)
    col = persona_action_column(bars)
    assert len(col) == len(bars)
    assert set(np.unique(col)).issubset({0.0, 1.0, 2.0})
    # Causal: a prefix produces the same actions (stateful strategy, per-bar).
    prefix = persona_action_column(bars.iloc[:100])
    np.testing.assert_array_equal(col.iloc[:100].to_numpy(), prefix.to_numpy())


def test_persona_actions_for_bars_matches_column():
    bars = make_bars(200)
    acts = persona_actions_for_bars(bars)
    col = persona_action_column(bars).to_numpy()
    np.testing.assert_array_equal(np.asarray(acts, dtype=float), col)
