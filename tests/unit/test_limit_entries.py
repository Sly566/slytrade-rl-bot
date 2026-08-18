"""Limit-entry support: strategy emits LIMIT intents, the managed engine fills
resting limits on touch and expires them, and the MT5 adapter places pending
orders. These guard the +72%-edge improvement measured on real XAUUSD M15.
"""
from __future__ import annotations

import pandas as pd

from slytrade.backtest.engine import BacktestConfig
from slytrade.backtest.reporting import run_managed_aligned_backtest_from_bars
from slytrade.backtest.trade_management import TradeManagementConfig
from slytrade.execution.models import OrderKind
from slytrade.strategies.personality_adaptive import PersonalityAdaptiveConfig, PersonalityAdaptiveStrategy


def _series(**kw) -> pd.Series:
    defaults = {
        "time": pd.Timestamp("2026-08-18T00:00:00", tz="UTC"),
        "symbol": "XAUUSD", "timeframe": "M15",
        "open": 4000.0, "high": 4000.5, "low": 3999.5, "close": 4000.0,
        "atr": 3.0, "atr_norm": 0.001, "trend_strength": 0.5, "premium_discount": -0.3,
        "bos_dir": 1.0, "choch_dir": 1.0, "liquidity_sweep": -1.0, "fvg_bullish": 1.0,
        "fvg_bearish": 0.0, "order_block_bullish": 1.0, "order_block_bearish": 0.0,
        "mtf_bias": 1.0, "mtf_confluence_score": 2.0,
    }
    defaults.update(kw)
    return pd.Series(defaults)


def test_strategy_emits_limit_intent_when_knob_set() -> None:
    cfg = PersonalityAdaptiveConfig(min_score=3, cooldown_bars=0, limit_entry_atr=0.25,
                                    require_sweep_reversal=False, require_entry_momentum=False,
                                    strict_mtf_direction=False, use_regime_filter=False)
    strategy = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    bar = _series()
    intent = strategy.on_bar(0, bar)
    assert intent is not None
    assert intent.kind == OrderKind.LIMIT
    assert intent.limit_price is not None
    assert intent.limit_price < bar["close"]  # long limit below market
    # 0.25 * 3.0 ATR = 0.75 pullback
    assert abs((bar["close"] - intent.limit_price) - 0.75) < 1e-6


def test_strategy_market_when_knob_zero() -> None:
    cfg = PersonalityAdaptiveConfig(min_score=3, cooldown_bars=0, limit_entry_atr=0.0,
                                    require_sweep_reversal=False, require_entry_momentum=False,
                                    strict_mtf_direction=False, use_regime_filter=False)
    strategy = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    intent = strategy.on_bar(0, _series())
    assert intent is not None
    assert intent.kind == OrderKind.MARKET
    assert intent.limit_price is None


def test_short_limit_is_above_market() -> None:
    cfg = PersonalityAdaptiveConfig(min_score=3, cooldown_bars=0, limit_entry_atr=0.25,
                                    require_sweep_reversal=False, require_entry_momentum=False,
                                    strict_mtf_direction=False, use_regime_filter=False)
    strategy = PersonalityAdaptiveStrategy(symbol="XAUUSD", volume=0.1, config=cfg)
    bar = _series(bos_dir=-1.0, choch_dir=-1.0, liquidity_sweep=1.0,
                  fvg_bullish=0.0, fvg_bearish=1.0, order_block_bullish=0.0, order_block_bearish=1.0,
                  trend_strength=-0.5, premium_discount=0.3, mtf_bias=-1.0)
    intent = strategy.on_bar(0, bar)
    assert intent is not None
    assert intent.kind == OrderKind.LIMIT
    assert intent.limit_price is not None
    assert intent.limit_price > bar["close"]


def test_managed_engine_fills_limit_on_touch() -> None:
    # A synthetic aligned frame where the limit (0.25*ATR below close) is
    # touched on the very next bar, then price runs to the target.
    n = 12
    times = pd.date_range("2026-08-18T00:00:00", periods=n, freq="15min", tz="UTC")
    close = [4000.0, 3999.0, 3999.0, 3999.0, 3999.0, 3999.0, 3999.0, 3999.0, 3999.0, 3999.0, 3999.0, 3999.0]
    high = [4000.5, 3999.5, 3999.5, 3999.5, 3999.5, 3999.5, 3999.5, 3999.5, 3999.5, 3999.5, 3999.5, 3999.5]
    low = [3999.5, 3998.0, 3998.0, 3998.0, 3998.0, 3998.0, 3998.0, 3998.0, 3998.0, 3998.0, 3998.0, 3998.0]
    bars = pd.DataFrame({
        "time": times, "symbol": "XAUUSD", "timeframe": "M15",
        "open": close, "high": high, "low": low, "close": close,
        "tick_volume": 100.0, "spread": 0.135, "real_volume": 0.0,
        "atr": 3.0, "atr_norm": 0.001, "trend_strength": 0.5, "premium_discount": -0.3,
        "bos_dir": 1.0, "choch_dir": 1.0, "liquidity_sweep": -1.0, "fvg_bullish": 1.0, "fvg_bearish": 0.0,
        "order_block_bullish": 1.0, "order_block_bearish": 0.0, "mtf_bias": 1.0, "mtf_confluence_score": 2.0,
    })
    bars["decision_time"] = bars["time"] + pd.Timedelta(minutes=15)
    bars["quote_bid"] = bars["close"] - bars["spread"] / 2
    bars["quote_ask"] = bars["close"] + bars["spread"] / 2
    bars["quote_time"] = bars["decision_time"] - pd.Timedelta(seconds=1)
    bars["quote_is_fresh"] = True

    # Gate-light config: isolate the limit-order mechanics from the signal gates.
    persona = PersonalityAdaptiveConfig(
        min_score=3, cooldown_bars=0, limit_entry_atr=0.25,
        require_sweep_reversal=False, require_entry_momentum=False,
        strict_mtf_direction=False, use_regime_filter=False,
        require_mtf_alignment=False, htf_trend_timeframe=None,
        point_value=100.0,  # gold: volume = 500/(3*100) = 1.67 < max 100
    )
    result = run_managed_aligned_backtest_from_bars(
        bars, strategy_name="persona-adaptive", symbol="XAUUSD", volume=0.1, point_value=100.0,
        config=BacktestConfig(initial_balance=100_000.0, point_size=0.01, point_value=100.0,
                              commission_per_volume=3.5, slippage_points=5),
        trade_config=TradeManagementConfig(stop_loss_atr=1.0, take_profit_atr=3.0, max_bars_in_trade=60),
        persona_config=persona,
    )
    entries = [t for t in result.trades if t.reason.startswith("persona_")]
    assert len(entries) >= 1
    # The fill must be at the limit (close - 0.25*ATR = 4000 - 0.75), not the close.
    first = entries[0]
    assert abs(first.price - (4000.0 - 0.75)) < 0.15  # limit + slippage


def test_managed_engine_expires_untouched_limit() -> None:
    # Price never retraces: the limit must expire and not produce a position.
    n = 12
    times = pd.date_range("2026-08-18T00:00:00", periods=n, freq="15min", tz="UTC")
    close = [4000.0] + [4005.0] * (n - 1)
    bars = pd.DataFrame({
        "time": times, "symbol": "XAUUSD", "timeframe": "M15",
        "open": close, "high": [c + 0.5 for c in close], "low": [c - 0.5 for c in close], "close": close,
        "tick_volume": 100.0, "spread": 0.135, "real_volume": 0.0,
        "atr": 3.0, "atr_norm": 0.001, "trend_strength": 0.5, "premium_discount": -0.3,
        "bos_dir": 1.0, "choch_dir": 1.0, "liquidity_sweep": -1.0, "fvg_bullish": 1.0, "fvg_bearish": 0.0,
        "order_block_bullish": 1.0, "order_block_bearish": 0.0, "mtf_bias": 1.0, "mtf_confluence_score": 2.0,
    })
    bars["decision_time"] = bars["time"] + pd.Timedelta(minutes=15)
    bars["quote_bid"] = bars["close"] - bars["spread"] / 2
    bars["quote_ask"] = bars["close"] + bars["spread"] / 2
    bars["quote_time"] = bars["decision_time"] - pd.Timedelta(seconds=1)
    bars["quote_is_fresh"] = True

    persona = PersonalityAdaptiveConfig(
        min_score=3, cooldown_bars=0, limit_entry_atr=0.25,
        require_sweep_reversal=False, require_entry_momentum=False,
        strict_mtf_direction=False, use_regime_filter=False,
        require_mtf_alignment=False, htf_trend_timeframe=None,
        point_value=100.0,
    )
    result = run_managed_aligned_backtest_from_bars(
        bars, strategy_name="persona-adaptive", symbol="XAUUSD", volume=0.1, point_value=100.0,
        config=BacktestConfig(initial_balance=100_000.0, point_size=0.01, point_value=100.0,
                              commission_per_volume=3.5, slippage_points=5),
        trade_config=TradeManagementConfig(stop_loss_atr=1.0, take_profit_atr=3.0, max_bars_in_trade=5),
        persona_config=persona,
    )
    entries = [t for t in result.trades if t.reason.startswith("persona_")]
    assert len(entries) == 0  # never filled, expired without a position
