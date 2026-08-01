import pandas as pd

from slytrade.backtest.engine import BacktestConfig
from slytrade.backtest.reporting import run_managed_aligned_backtest_from_bars
from slytrade.execution.models import Side
from slytrade.strategies.baselines import ICTConfluenceStrategy


def base_bar(**overrides):
    row = {
        "close": 100.0,
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


def make_aligned_bars() -> pd.DataFrame:
    rows = []
    for index in range(8):
        rows.append(
            {
                "time": pd.Timestamp("2026-07-01T00:00:00Z") + pd.Timedelta(minutes=index),
                "decision_time": pd.Timestamp("2026-07-01T00:01:00Z") + pd.Timedelta(minutes=index),
                "symbol": "XAUUSD",
                "timeframe": "M1",
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "atr": 1.0,
                "quote_time": pd.Timestamp("2026-07-01T00:01:00Z") + pd.Timedelta(minutes=index),
                "quote_bid": 100.0 + index,
                "quote_ask": 100.2 + index,
                "quote_mid": 100.1 + index,
                "quote_spread": 0.2,
                "quote_age_seconds": 0.0,
                "quote_is_fresh": True,
                "tick_mid_high": 101.0 + index,
                "tick_mid_low": 99.0 + index,
                "bos_dir": 1.0 if index == 0 else 0.0,
                "choch_dir": 0.0,
                "liquidity_sweep": -1.0 if index == 0 else 0.0,
                "fvg_bullish": 1.0 if index == 0 else 0.0,
                "fvg_bearish": 0.0,
                "order_block_bullish": 0.0,
                "order_block_bearish": 0.0,
                "premium_discount": -0.5,
                "trend_strength": 0.5,
                "tick_rate_per_second": 2.0,
                "session_london": 1.0,
                "session_ny_am": 0.0,
                "session_ny_pm": 0.0,
                "session_asia": 0.0,
                "session_other": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_ict_confluence_long_signal():
    strategy = ICTConfluenceStrategy(symbol="XAUUSD", volume=0.1, min_score=4)
    intent = strategy.on_bar(0, base_bar())

    assert intent is not None
    assert intent.side == Side.BUY
    assert intent.reason.startswith("ict_confluence_long")


def test_ict_confluence_requires_allowed_session():
    strategy = ICTConfluenceStrategy(symbol="XAUUSD", volume=0.1, min_score=4, allowed_sessions=("ny_am",))
    intent = strategy.on_bar(0, base_bar(session_london=1.0, session_ny_am=0.0))

    assert intent is None


def test_ict_confluence_cooldown():
    strategy = ICTConfluenceStrategy(symbol="XAUUSD", volume=0.1, min_score=4, cooldown_bars=10)
    first = strategy.on_bar(0, base_bar())
    second = strategy.on_bar(1, base_bar())

    assert first is not None
    assert second is None


def test_ict_confluence_managed_backtest_runs():
    result = run_managed_aligned_backtest_from_bars(
        make_aligned_bars(),
        strategy_name="ict-confluence",
        volume=0.1,
        config=BacktestConfig(initial_balance=100_000, point_value=1.0),
    )

    assert len(result.orders) >= 1
