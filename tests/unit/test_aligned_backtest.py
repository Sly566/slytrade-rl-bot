import numpy as np
import pandas as pd
from typer.testing import CliRunner

from slytrade.backtest.aligned_engine import AlignedBacktestEngine, quote_from_aligned_bar
from slytrade.backtest.engine import BacktestConfig, BuyAndHoldOnceStrategy
from slytrade.backtest.reporting import run_aligned_backtest_from_bars
from slytrade.cli import app
from slytrade.data.alignment import align_market_data


def make_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-01", periods=4, freq="min", tz="UTC"),
            "symbol": "XAUUSDm",
            "timeframe": "M1",
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "tick_volume": [10, 10, 10, 10],
            "spread": [10, 10, 10, 10],
            "real_volume": [0, 0, 0, 0],
        }
    )


def make_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-01T00:01:00Z", periods=4, freq="min").floor("s"),
            "time_msc": pd.date_range("2026-07-01T00:01:00Z", periods=4, freq="min"),
            "symbol": "XAUUSD",
            "bid": [100.0, 101.0, 102.0, 103.0],
            "ask": [100.2, 101.2, 102.2, 103.2],
            "last": np.zeros(4),
            "volume": np.ones(4),
            "volume_real": np.zeros(4),
            "flags": np.zeros(4),
            "spread": [0.2, 0.2, 0.2, 0.2],
            "mid": [100.1, 101.1, 102.1, 103.1],
        }
    )


def test_align_market_data_attaches_decision_quotes():
    dataset = align_market_data(make_bars(), make_ticks(), timeframe="M1")

    assert "quote_bid" in dataset.bars.columns
    assert "quote_ask" in dataset.bars.columns
    assert "quote_age_seconds" in dataset.bars.columns
    assert bool(dataset.bars.loc[0, "quote_is_fresh"])
    assert dataset.bars.loc[0, "quote_ask"] == 100.2


def test_quote_from_aligned_bar():
    dataset = align_market_data(make_bars(), make_ticks(), timeframe="M1")
    quote = quote_from_aligned_bar(dataset.bars.iloc[0])

    assert quote is not None
    assert quote.bid == 100.0
    assert quote.ask == 100.2


def test_aligned_backtest_engine_runs_without_ticks_scan():
    dataset = align_market_data(make_bars(), make_ticks(), timeframe="M1")
    engine = AlignedBacktestEngine(BacktestConfig(initial_balance=100_000, point_value=1.0))
    result = engine.run(dataset.bars, BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=1.0))

    assert result.metrics.trades == 1
    assert result.trades[0].price == 100.2


def test_run_aligned_backtest_from_reporting_helper():
    dataset = align_market_data(make_bars(), make_ticks(), timeframe="M1")
    result = run_aligned_backtest_from_bars(dataset.bars, strategy_name="buy-and-hold", volume=1.0)

    assert result.metrics.trades == 1


def test_run_aligned_backtest_cli(tmp_path):
    dataset = align_market_data(make_bars(), make_ticks(), timeframe="M1")
    bars_path = tmp_path / "aligned_bars.csv"
    dataset.bars.to_csv(bars_path, index=False)
    runner = CliRunner()

    result = runner.invoke(app, ["run-aligned-backtest", "--bars-file", str(bars_path), "--strategy", "buy-and-hold"])

    assert result.exit_code == 0
    assert "Backtest Report" in result.stdout
