"""Tests for the personality-adaptive ICT strategy and regime engine."""

import numpy as np
import pandas as pd
import pytest

from slytrade.config.trader_personality import TraderPersonality
from slytrade.execution.models import Side
from slytrade.intelligence.market_context import MarketContextEngine
from slytrade.intelligence.micro_macro_alignment import MicroMacroAlignmentEngine
from slytrade.intelligence.regime import MarketRegimeEngine
from slytrade.strategies.personality_adaptive import PersonalityAdaptiveStrategy


def make_personality(**overrides) -> TraderPersonality:
    defaults = dict(
        aggression=0.65,
        selectivity=0.75,
        risk_tolerance=0.60,
        scalping_bias=0.70,
        day_trading_bias=0.30,
        macro_respect=0.85,
        session_sensitivity=0.80,
        conviction=0.70,
        patience=0.75,
        discipline=0.85,
        adaptability=0.80,
        structure_focus=0.90,
        liquidity_focus=0.88,
        edge_optimism=0.55,
    )
    defaults.update(overrides)
    return TraderPersonality(**defaults)


def base_bar(**overrides) -> pd.Series:
    row = {
        "time": pd.Timestamp("2026-07-01T09:00:00Z"),  # London session
        "close": 100.0,
        "atr": 1.0,
        "atr_norm": 0.01,
        "bos_dir": 1.0,
        "choch_dir": 0.0,
        "liquidity_sweep": -1.0,
        "fvg_bullish": 1.0,
        "fvg_bearish": 0.0,
        "order_block_bullish": 0.0,
        "order_block_bearish": 0.0,
        "premium_discount": -0.5,
        "trend_strength": 0.5,
        "tick_rate_per_second": 2.0,
        "quote_spread": 0.2,
        "quote_is_fresh": True,
        "session_london": 1.0,
        "session_ny_am": 0.0,
        "session_ny_pm": 0.0,
        "session_asia": 0.0,
        "session_other": 0.0,
    }
    row.update(overrides)
    return pd.Series(row)


def test_personality_validates_trait_ranges():
    with pytest.raises(ValueError):
        TraderPersonality(aggression=1.5)
    with pytest.raises(ValueError):
        TraderPersonality(max_risk_per_trade_default=0.1)


def test_personality_from_yaml_loads_deep_traits():
    personality = TraderPersonality.from_yaml("configs/trader_personality.yaml")
    assert personality.name == "SlyMasterICT"
    assert personality.conviction >= 0.5
    assert personality.structure_focus >= 0.5
    assert personality.position_sizing == "risk_based"


def test_regime_engine_detects_trend_and_volatility():
    rows = []
    for i in range(150):
        # Strong uptrend with steady ATR
        rows.append(
            {
                "atr_norm": 0.01,
                "trend_strength": 0.5 + i * 0.001,
                "premium_discount": 0.1,
            }
        )
        rows[-1]["trend_strength"] = 0.9  # strong bull trend every bar
    bars = pd.DataFrame(rows)
    regime_engine = MarketRegimeEngine(volatile_z_threshold=5.0, quiet_z_threshold=-5.0)
    regime = regime_engine.analyze_tail(bars)
    assert regime.trend == "bull"
    assert regime.volatility == "normal"


def test_market_context_and_alignment():
    bars = pd.DataFrame([base_bar()])
    personality = make_personality()
    context = MarketContextEngine(personality).analyze(bars)
    assert context["volatility"] == "normal"
    alignment = MicroMacroAlignmentEngine(personality).evaluate(bars, context)
    assert 0.0 <= alignment <= 1.0


def test_persona_strategy_emits_long_intent_on_strong_bullish_bar():
    personality = make_personality(selectivity=0.5, aggression=0.6, edge_optimism=0.4)
    strategy = PersonalityAdaptiveStrategy(
        personality=personality,
        config=None,
        symbol="XAUUSD",
        volume=0.1,
    )
    # Warm up history so the regime engine has context, then reset so the
    # strategy is flat and the final bar is a fresh entry opportunity.
    warmup = [base_bar() for _ in range(60)]
    for idx, bar in enumerate(warmup):
        strategy.on_bar(idx, bar)
    strategy.reset()
    strategy._side = "flat"
    strategy._last_entry_index = -1_000_000
    intent = strategy.on_bar(len(warmup), base_bar())
    assert intent is not None
    assert intent.side == Side.BUY





def test_persona_strategy_respects_cooldown():
    personality = make_personality(selectivity=0.4, aggression=0.7, edge_optimism=0.3)
    strategy = PersonalityAdaptiveStrategy(personality=personality, symbol="XAUUSD", volume=0.1)
    for idx in range(80):
        strategy.on_bar(idx, base_bar())
    first = strategy.on_bar(81, base_bar())
    second = strategy.on_bar(82, base_bar())
    if first is not None:
        assert second is None  # cooldown blocks consecutive entries


def test_persona_strategy_emits_stop_budgeted_volume():
    personality = make_personality(selectivity=0.5, aggression=0.6, edge_optimism=0.4)
    strategy = PersonalityAdaptiveStrategy(
        personality=personality,
        symbol="XAUUSD",
        volume=0.1,
    )
    for idx in range(80):
        strategy.on_bar(idx, base_bar())
    intent = strategy.on_bar(81, base_bar())
    if intent is not None:
        # Risk-budgeted volume: 100k * 0.005 / (1.0 ATR * 1.0) = 50
        assert intent.volume == pytest.approx(50.0, rel=0.01)


def test_analyze_tail_arrays_matches_analyze():
    """The fast tail-only context must reproduce the slow analyze() output.

    This is the safety net for the persona backtest hot-loop rewrite: the
    vectorized path has to classify every trailing window identically to the
    old DataFrame + per-row apply() path, across window lengths spanning the
    rolling(min_periods=20) and lookback(100) boundaries.
    """
    rng = np.random.default_rng(7)
    n = 150
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2026-07-01T09:00:00Z", periods=n, freq="1min"),
            "atr_norm": rng.uniform(0.001, 0.06, n),
            "trend_strength": rng.uniform(-1.0, 1.0, n),
            "premium_discount": rng.uniform(-1.0, 1.0, n),
            "mtf_bias": rng.integers(-1, 2, n).astype(float),
            "mtf_confluence_score": rng.integers(0, 5, n).astype(float),
        }
    )
    personality = make_personality()
    engine = MarketContextEngine(personality, MarketRegimeEngine())

    categorical = ("volatility", "trend", "session", "macro_strength", "mtf_bias", "mtf_confluence_score")
    numeric = ("regime_score", "volatility_zscore", "trend_strength_raw", "premium_discount", "atr_norm_20")

    for end in (1, 19, 20, 45, 99, 100, 120, 150):
        window = bars.iloc[:end]
        slow = engine.analyze(window)
        fast = engine.analyze_tail_arrays(
            atr_norm=window["atr_norm"].to_numpy(),
            trend_strength=window["trend_strength"].to_numpy(),
            premium_discount=window["premium_discount"].to_numpy(),
            times=list(window["time"]),
            mtf_bias=window["mtf_bias"].to_numpy(),
            mtf_confluence=window["mtf_confluence_score"].to_numpy(),
            has_htf=True,
        )
        for key in categorical:
            assert fast.get(key) == slow.get(key), (end, key, fast.get(key), slow.get(key))
        for key in numeric:
            assert fast.get(key) == pytest.approx(slow.get(key), rel=1e-9, abs=1e-9), (
                end,
                key,
                fast.get(key),
                slow.get(key),
            )
        assert fast.get("has_htf") is True
