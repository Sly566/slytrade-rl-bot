import numpy as np
import pandas as pd
from typer.testing import CliRunner

from slytrade.backtest.engine import BacktestConfig, BuyAndHoldOnceStrategy
from slytrade.backtest.reporting import load_ticks_file, run_tick_backtest_from_frames
from slytrade.backtest.tick_engine import TickBacktestEngine, quote_from_tick
from slytrade.cli import app
from slytrade.data.sample_generator import generate_sample_bars, generate_sample_ticks
from slytrade.data.time import parse_utc_datetime
from slytrade.execution.models import OrderStatus


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


def make_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC").floor("s"),
            "time_msc": pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "bid": [99.0, 100.0, 101.0, 102.0],
            "ask": [101.0, 102.0, 103.0, 104.0],
            "last": np.zeros(4),
            "volume": np.ones(4),
            "volume_real": np.zeros(4),
            "flags": np.zeros(4),
            "spread": [2.0, 2.0, 2.0, 2.0],
            "mid": [100.0, 101.0, 102.0, 103.0],
        }
    )


def test_quote_from_tick_uses_bid_ask():
    quote = quote_from_tick(make_ticks().iloc[0])

    assert quote.bid == 99.0
    assert quote.ask == 101.0
    assert quote.mid == 100.0


def test_tick_backtest_engine_executes_on_tick_quote():
    engine = TickBacktestEngine(BacktestConfig(initial_balance=100_000, point_size=0.01, point_value=1.0))
    result = engine.run(make_bars(), make_ticks(), BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=1.0))

    assert len(result.trades) == 1
    assert result.trades[0].price == 101.0
    assert result.reports[0].status == OrderStatus.FILLED
    assert result.metrics.final_equity > result.metrics.start_equity


def test_tick_backtest_from_reporting_helper():
    result = run_tick_backtest_from_frames(
        make_bars(),
        make_ticks(),
        strategy_name="buy-and-hold",
        volume=1.0,
        config=BacktestConfig(initial_balance=100_000, point_size=0.01, point_value=1.0),
    )

    assert result.metrics.trades == 1


def test_load_ticks_file_csv(tmp_path):
    path = tmp_path / "ticks.csv"
    make_ticks().to_csv(path, index=False)

    loaded = load_ticks_file(path)

    assert len(loaded) == 4
    assert "bid" in loaded.columns


def test_run_tick_backtest_cli_with_generated_samples(tmp_path):
    bars_path = tmp_path / "bars.csv"
    ticks_path = tmp_path / "ticks.csv"
    start = parse_utc_datetime("2026-01-01")
    generate_sample_bars(start=start, periods=50).to_csv(bars_path, index=False)
    generate_sample_ticks(start=start, periods=200).to_csv(ticks_path, index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-tick-backtest",
            "--bars-file",
            str(bars_path),
            "--ticks-file",
            str(ticks_path),
            "--strategy",
            "buy-and-hold",
        ],
    )

    assert result.exit_code == 0
    assert "Backtest Report" in result.stdout
