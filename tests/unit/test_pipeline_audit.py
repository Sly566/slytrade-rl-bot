"""Truthfulness audit regression tests.

Lock in the findings of the pipeline audit:
- ATR matches canonical Wilder(14) in steady state (warmup is a flat-mean fill).
- The spread gate uses PRICE units (the old max_spread_points compared price to
  points and could never trigger correctly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.features.ict import compute_atr
from slytrade.strategies.personality_adaptive import PersonalityAdaptiveConfig, PersonalityAdaptiveStrategy


def make_bars(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    high = close + rng.uniform(0.01, 0.4, n)
    low = close - rng.uniform(0.01, 0.4, n)
    open_ = close - rng.uniform(-0.2, 0.2, n)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "symbol": "XAUUSD", "timeframe": "M1",
            "open": open_, "high": high, "low": low, "close": close,
            "tick_volume": rng.uniform(50, 300, n), "spread": 2.0, "real_volume": 0.0,
        }
    )


def _canonical_wilder_atr(high, low, close, period=14):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def test_atr_matches_canonical_wilder_in_steady_state():
    bars = make_bars(500, seed=11)
    got = compute_atr(bars).to_numpy(float)
    canonical = _canonical_wilder_atr(
        bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float)
    )
    # Steady state (well past the warmup) must be exact.
    np.testing.assert_allclose(got[100:], canonical[100:], rtol=0, atol=1e-9)
    # Warmup is a flat mean fill (bars 0..period-1 share one value) — a
    # legitimate, documented choice that converges at bar `period`.
    assert np.isclose(got[13], canonical[13], rtol=0, atol=1e-9)


def test_spread_gate_blocks_wide_spread_entry():
    from slytrade.execution.models import Side

    cfg = PersonalityAdaptiveConfig(
        require_sweep_reversal=False, require_entry_momentum=False,
        strict_mtf_direction=False, htf_trend_timeframe=None, max_spread=0.30,
    )
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    wide = pd.Series({"quote_spread": 0.50, "open": 100.0, "close": 101.0, "atr": 1.0,
                      "bos_dir": 1.0, "liquidity_sweep": -1.0, "fvg_bullish": 1.0,
                      "order_block_bullish": 1.0, "premium_discount": -0.5, "trend_strength": 0.5})
    assert s._setup_quality_ok(Side.BUY, wide) is False
    tight = wide.copy()
    tight["quote_spread"] = 0.10
    assert s._setup_quality_ok(Side.BUY, tight) is True
