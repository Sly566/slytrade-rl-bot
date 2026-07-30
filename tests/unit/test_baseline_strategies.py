import numpy as np
import pandas as pd

from slytrade.backtest.engine import BacktestConfig, BarBacktestEngine
from slytrade.execution.models import Side
from slytrade.strategies.baselines import (
    BuyAndHoldStrategy,
    ICTBiasBaselineStrategy,
    MovingAverageCrossStrategy,
    NoTradeStrategy,
)


def make_bars(close: list[float] | np.ndarray) -> pd.DataFrame:
    close_arr = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(close_arr), freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": close_arr,
            "high": close_arr + 0.5,
            "low": close_arr - 0.5,
            "close": close_arr,
            "tick_volume": np.full(len(close_arr), 100),
            "spread": np.full(len(close_arr), 5),
            "real_volume": np.zeros(len(close_arr)),
        }
    )


def test_no_trade_strategy_returns_none():
    strategy = NoTradeStrategy()
    bars = make_bars([1, 2, 3])

    assert strategy.on_bar(0, bars.iloc[0]) is None


def test_no_trade_backtest_has_no_trades():
    engine = BarBacktestEngine(BacktestConfig(initial_balance=100_000))
    result = engine.run(make_bars([100, 101, 102]), NoTradeStrategy())

    assert result.metrics.trades == 0
    assert result.trades == []
    assert result.orders == []


def test_buy_and_hold_submits_once():
    strategy = BuyAndHoldStrategy(symbol="XAUUSD", volume=0.1)
    bars = make_bars([100, 101, 102])

    first = strategy.on_bar(0, bars.iloc[0])
    second = strategy.on_bar(1, bars.iloc[1])

    assert first is not None
    assert first.side == Side.BUY
    assert second is None


def test_moving_average_cross_generates_signal():
    strategy = MovingAverageCrossStrategy(symbol="XAUUSD", volume=0.1, fast_window=2, slow_window=3)
    bars = make_bars([5, 4, 3, 4, 5, 6])
    intents = [strategy.on_bar(index, row) for index, row in bars.iterrows()]

    assert any(intent is not None and intent.side == Side.BUY for intent in intents)


def test_moving_average_cross_backtest_runs_through_paper_path():
    engine = BarBacktestEngine(BacktestConfig(initial_balance=100_000, point_size=0.01, point_value=1.0))
    strategy = MovingAverageCrossStrategy(symbol="XAUUSD", volume=0.1, fast_window=2, slow_window=3)
    result = engine.run(make_bars([5, 4, 3, 4, 5, 6, 7]), strategy)

    assert result.metrics.trades >= 1
    assert len(result.orders) >= 1
    assert len(result.trades) >= 1


def test_ict_bias_baseline_long_and_short_signals():
    strategy = ICTBiasBaselineStrategy(symbol="XAUUSD", volume=0.1)
    long_bar = pd.Series({"close": 100.0, "bos_dir": 1.0, "choch_dir": 0.0, "premium_discount": -0.5})
    short_bar = pd.Series({"close": 99.0, "bos_dir": -1.0, "choch_dir": 0.0, "premium_discount": 0.5})

    long_intent = strategy.on_bar(0, long_bar)
    short_intent = strategy.on_bar(1, short_bar)

    assert long_intent is not None
    assert long_intent.side == Side.BUY
    assert short_intent is not None
    assert short_intent.side == Side.SELL


def test_ict_bias_baseline_backtest_with_feature_columns():
    bars = make_bars([100, 101, 102, 101, 100])
    bars["bos_dir"] = [0.0, 1.0, 0.0, -1.0, 0.0]
    bars["choch_dir"] = 0.0
    bars["premium_discount"] = [-0.5, -0.5, -0.2, 0.5, 0.2]
    bars["liquidity_sweep"] = 0.0
    engine = BarBacktestEngine(BacktestConfig(initial_balance=100_000, point_size=0.01, point_value=1.0))

    result = engine.run(bars, ICTBiasBaselineStrategy(symbol="XAUUSD", volume=0.1))

    assert result.metrics.trades >= 1
    assert len(result.orders) >= 1
