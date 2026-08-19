"""Dynamic (score-weighted) gate model tests.

The quality gates must not hard-stop a strong confluence setup: each failed
gate subtracts ``gate_penalty`` from the raw score, and the bot executes when
(score - penalties) >= threshold. The spread gate remains a hard cost-control.
"""
from __future__ import annotations

from collections import deque

import pandas as pd

from slytrade.execution.models import Side
from slytrade.strategies.personality_adaptive import PersonalityAdaptiveConfig, PersonalityAdaptiveStrategy


def _neutral_bar(**overrides) -> pd.Series:
    row = {
        "time": pd.Timestamp("2026-07-01T09:00:00Z"),
        "close": 100.0, "open": 100.0, "atr": 1.0, "atr_norm": 0.01,
        "bos_dir": 0.0, "choch_dir": 0.0, "liquidity_sweep": 0.0,
        "fvg_bullish": 0.0, "fvg_bearish": 0.0,
        "order_block_bullish": 0.0, "order_block_bearish": 0.0,
        "premium_discount": 0.0, "trend_strength": 0.0,
        "quote_spread": 0.2, "quote_is_fresh": True,
        "session_london": 1.0, "mtf_bias": 0.0, "mtf_confluence_score": 0.0,
        "htf_h4_trend_strength": 1.0,
    }
    row.update(overrides)
    return pd.Series(row)


def _bullish(**overrides) -> pd.Series:
    row = dict(
        close=101.0, open=100.5,
        bos_dir=1.0, liquidity_sweep=-1.0, fvg_bullish=1.0,
        order_block_bullish=1.0, premium_discount=-0.5, trend_strength=0.5,
    )
    row.update(overrides)
    return _neutral_bar(**row)


def _base_config(**overrides) -> PersonalityAdaptiveConfig:
    kwargs = dict(
        require_sweep_reversal=False,
        require_entry_momentum=True,
        strict_mtf_direction=False,
        htf_trend_timeframe=None,
        use_regime_filter=False,
        require_mtf_alignment=False,
    )
    kwargs.update(overrides)
    return PersonalityAdaptiveConfig(**kwargs)


def _warm(s, n=60):
    for i in range(n):
        s.on_bar(i, _neutral_bar())


def test_dynamic_lets_strong_setup_override_momentum() -> None:
    cfg = _base_config(dynamic_gates=True, gate_penalty=2.0)
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    _warm(s)
    # Strong bullish structure (score ~7) but momentum fails (close < open).
    intent = s.on_bar(100, _bullish(close=100.0, open=101.0, mtf_bias=1.0))
    assert intent is not None, "a score-7 setup must override a momentum miss in dynamic mode"
    assert intent.side.value == "buy"


def test_hard_gates_still_block_same_setup() -> None:
    cfg = _base_config(dynamic_gates=False)
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    _warm(s)
    intent = s.on_bar(100, _bullish(close=100.0, open=101.0, mtf_bias=1.0))
    assert intent is None, "legacy hard gates block on momentum regardless of score"


def test_dynamic_still_blocks_marginal_setup() -> None:
    cfg = _base_config(dynamic_gates=True, gate_penalty=2.0)
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    _warm(s)
    # Score ~4 (BOS + premium + trend only) with momentum fail → 4-2=2 < threshold.
    bar = _bullish(close=100.0, open=101.0, liquidity_sweep=0.0, fvg_bullish=0.0, order_block_bullish=0.0)
    intent = s.on_bar(100, bar)
    assert intent is None, "a marginal setup with a gate miss must still be refused"


def test_spread_gate_stays_hard_in_dynamic_mode() -> None:
    cfg = _base_config(dynamic_gates=True, gate_penalty=0.5, max_spread=0.5)
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    _warm(s)
    bar = _bullish(quote_spread=1.0, mtf_bias=1.0)
    intent = s.on_bar(100, bar)
    assert intent is None, "wide spread is a cost control and must always hard-block"


def test_gate_penalties_count_each_failed_gate() -> None:
    cfg = _base_config(dynamic_gates=True, gate_penalty=2.0, strict_mtf_direction=True)
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    s._has_mtf_bias = True
    s._mtf_bias = deque([-1.0])  # opposes a long
    bar = _bullish(close=100.0, open=101.0)  # momentum fails
    context = {"volatility": "normal", "trend": "bull", "session": "london", "has_htf": True, "regime_score": 0.5}
    penalty = s._gate_penalties(Side.BUY, bar, context, alignment=0.9)
    assert penalty == 4.0  # momentum (-2) + strict MTF (-2)


def test_gate_penalties_zero_when_all_green() -> None:
    cfg = _base_config(dynamic_gates=True, gate_penalty=2.0, strict_mtf_direction=True)
    s = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    s._has_mtf_bias = True
    s._mtf_bias = deque([1.0])  # supports a long
    bar = _bullish()  # momentum ok (close > open)
    context = {"volatility": "normal", "trend": "bull", "session": "london", "has_htf": True, "regime_score": 0.5}
    penalty = s._gate_penalties(Side.BUY, bar, context, alignment=0.9)
    assert penalty == 0.0
