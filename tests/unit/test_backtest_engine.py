import pandas as pd

from slytrade.backtest.engine import BacktestConfig, BarBacktestEngine, BuyAndHoldOnceStrategy, quote_from_bar


def make_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "tick_volume": [10, 10, 10, 10],
            "spread": [10, 10, 10, 10],
            "real_volume": [0, 0, 0, 0],
        }
    )


def test_quote_from_bar_uses_spread_points():
    quote = quote_from_bar(make_bars().iloc[0], default_spread_points=20, point_size=0.01)

    assert quote.bid == 99.95
    assert quote.ask == 100.05


def test_bar_backtest_engine_runs_buy_and_hold():
    engine = BarBacktestEngine(BacktestConfig(initial_balance=100_000, point_size=0.01, point_value=1.0))
    strategy = BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=1.0)

    result = engine.run(make_bars(), strategy)

    assert len(result.equity_curve) == 5
    assert result.metrics.trades == 1
    assert result.metrics.final_equity > result.metrics.start_equity
