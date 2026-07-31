import pandas as pd

from slytrade.backtest.engine import BacktestConfig, BuyAndHoldOnceStrategy
from slytrade.backtest.tick_engine import TickBacktestEngine, is_quote_fresh, quote_age_seconds, quote_from_tick
from slytrade.data.diagnostics import inspect_tick_bar_coverage


def make_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "tick_volume": [10, 10, 10],
            "spread": [10, 10, 10],
            "real_volume": [0, 0, 0],
        }
    )


def make_sparse_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "time_msc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "symbol": ["XAUUSD"],
            "bid": [100.0],
            "ask": [100.2],
            "last": [0.0],
            "volume": [1.0],
            "volume_real": [0.0],
            "flags": [0.0],
            "spread": [0.2],
            "mid": [100.1],
        }
    )


def test_quote_age_and_freshness():
    quote = quote_from_tick(make_sparse_ticks().iloc[0])
    decision_time = pd.Timestamp("2026-01-01T00:00:04Z")

    assert quote_age_seconds(quote, decision_time) == 4.0
    assert is_quote_fresh(quote, decision_time, 5.0)
    assert not is_quote_fresh(quote, decision_time, 3.0)


def test_coverage_reports_stale_quotes():
    coverage = inspect_tick_bar_coverage(make_bars(), make_sparse_ticks(), max_quote_age_seconds=5.0)

    assert coverage.bars == 3
    assert coverage.bars_with_tick_before_decision == 3
    assert coverage.bars_with_fresh_tick_before_decision == 0
    assert coverage.bars_with_stale_tick_before_decision == 3
    assert coverage.first_stale_decision_time == "2026-01-01 00:01:00+00:00"


def test_tick_backtest_skips_execution_when_quotes_are_stale_and_fallback_disabled():
    engine = TickBacktestEngine(
        BacktestConfig(
            initial_balance=100_000,
            max_quote_age_seconds=5.0,
            allow_bar_quote_fallback=False,
        )
    )
    result = engine.run(make_bars(), make_sparse_ticks(), BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=1.0))

    assert result.metrics.trades == 0
    assert result.orders == []
    assert engine.last_stats.stale_quotes == 3
    assert engine.last_stats.fallback_bar_quotes == 0


def test_tick_backtest_can_fallback_to_bar_quote_when_enabled():
    engine = TickBacktestEngine(
        BacktestConfig(
            initial_balance=100_000,
            max_quote_age_seconds=5.0,
            allow_bar_quote_fallback=True,
        )
    )
    result = engine.run(make_bars(), make_sparse_ticks(), BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=1.0))

    assert result.metrics.trades == 1
    assert engine.last_stats.fallback_bar_quotes >= 1
