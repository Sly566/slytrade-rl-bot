import pandas as pd

from slytrade.backtest.engine import BacktestConfig, BarBacktestEngine, BuyAndHoldOnceStrategy
from slytrade.execution.models import OrderStatus


def make_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "tick_volume": [10, 10, 10, 10, 10],
            "spread": [10, 10, 10, 10, 10],
            "real_volume": [0, 0, 0, 0, 0],
        }
    )


def test_backtest_uses_paper_broker_oms_and_ledger():
    engine = BarBacktestEngine(BacktestConfig(initial_balance=100_000, point_size=0.01, point_value=1.0))
    result = engine.run(make_bars(), BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=1.0))

    assert len(result.orders) == 1
    assert len(result.trades) == 1
    assert result.reports[0].status == OrderStatus.FILLED
    assert result.orders[0].status == OrderStatus.FILLED
    assert result.trades[0].reason == "buy_and_hold_once"
    assert result.metrics.trades == 1
    assert result.metrics.final_equity > result.metrics.start_equity


def test_backtest_guardrail_rejection_is_recorded_as_order_but_not_trade():
    engine = BarBacktestEngine(
        BacktestConfig(
            initial_balance=100_000,
            point_size=0.01,
            point_value=1.0,
            max_spread_points=1.0,
        )
    )
    result = engine.run(make_bars(), BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=1.0))

    assert len(result.orders) == 1
    assert len(result.trades) == 0
    assert result.reports[0].status == OrderStatus.REJECTED
    assert "spread too high" in result.reports[0].message
    assert result.metrics.trades == 0
