"""Tests for the persona entry-quality filters (incl. HTF macro-trend gate)."""
from __future__ import annotations

import pandas as pd

from slytrade.strategies.personality_adaptive import PersonalityAdaptiveConfig, PersonalityAdaptiveStrategy


def neutral_bar(**overrides) -> pd.Series:
    """A bar with no confluence — builds context without triggering entries."""
    row = {
        "time": pd.Timestamp("2026-07-01T09:00:00Z"),
        "close": 100.0,
        "open": 100.0,
        "atr": 1.0,
        "atr_norm": 0.01,
        "bos_dir": 0.0,
        "choch_dir": 0.0,
        "liquidity_sweep": 0.0,
        "fvg_bullish": 0.0,
        "fvg_bearish": 0.0,
        "order_block_bullish": 0.0,
        "order_block_bearish": 0.0,
        "premium_discount": 0.0,
        "trend_strength": 0.0,
        "tick_rate_per_second": 2.0,
        "quote_spread": 0.2,
        "quote_is_fresh": True,
        "session_asia": 0.0,
        "session_london": 1.0,
        "session_ny_am": 0.0,
        "session_ny_pm": 0.0,
        "session_other": 0.0,
        "mtf_bias": 0.0,
        "mtf_confluence_score": 0.0,
        "htf_h4_trend_strength": 1.0,
    }
    row.update(overrides)
    return pd.Series(row)


def bullish_bar(**overrides) -> pd.Series:
    row = dict(
        close=101.0, open=100.5,  # green candle
        bos_dir=1.0, liquidity_sweep=-1.0, fvg_bullish=1.0,
        order_block_bullish=1.0, premium_discount=-0.5, trend_strength=0.5,
    )
    row.update(overrides)
    return neutral_bar(**row)


def bearish_bar(**overrides) -> pd.Series:
    row = dict(
        close=99.0, open=99.5,  # red candle
        bos_dir=-1.0, liquidity_sweep=1.0, fvg_bearish=1.0,
        order_block_bearish=1.0, premium_discount=0.5, trend_strength=-0.5,
        mtf_bias=-1.0,
    )
    row.update(overrides)
    return neutral_bar(**row)


def _warm(strategy, n=60):
    for i in range(n):
        strategy.on_bar(i, neutral_bar())


def test_htf_trend_gate_blocks_short_against_h4_uptrend():
    cfg = PersonalityAdaptiveConfig(
        require_sweep_reversal=False,
        require_entry_momentum=False,
        strict_mtf_direction=False,
        htf_trend_timeframe="h4",
        use_regime_filter=False,
    )
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    _warm(s)
    # H4 trend is up (+1) -> a short setup must be refused.
    intent = s.on_bar(100, bearish_bar(htf_h4_trend_strength=1.0))
    assert intent is None or intent.side.value != "sell"


def test_htf_trend_gate_allows_long_with_h4_uptrend():
    cfg = PersonalityAdaptiveConfig(
        require_sweep_reversal=False,
        require_entry_momentum=False,
        strict_mtf_direction=False,
        htf_trend_timeframe="h4",
        use_regime_filter=False,
    )
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    _warm(s)
    intent = s.on_bar(100, bullish_bar(htf_h4_trend_strength=1.0))
    assert intent is not None
    assert intent.side.value == "buy"


def test_htf_trend_gate_noop_when_column_absent():
    # Live bars may not carry htf_* columns; the gate must be a no-op (never a
    # full block) so live paper/demo trading is unaffected.
    cfg = PersonalityAdaptiveConfig(
        require_sweep_reversal=False,
        require_entry_momentum=False,
        strict_mtf_direction=False,
        htf_trend_timeframe="h4",
        use_regime_filter=False,
    )
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    _warm(s)
    bar = bullish_bar().drop("htf_h4_trend_strength")
    intent = s.on_bar(100, bar)
    assert intent is not None
    assert intent.side.value == "buy"


def test_entry_momentum_filter_blocks_fading():
    cfg = PersonalityAdaptiveConfig(
        require_sweep_reversal=False,
        require_entry_momentum=True,
        strict_mtf_direction=False,
        use_regime_filter=False,
    )
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    _warm(s)
    # Bullish footprint but a RED candle -> momentum gate blocks the long.
    intent = s.on_bar(100, bullish_bar(close=99.0, open=100.0))
    assert intent is None or intent.side.value != "buy"
