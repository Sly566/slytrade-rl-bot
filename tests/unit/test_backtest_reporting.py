import numpy as np
import pandas as pd
from typer.testing import CliRunner

from slytrade.backtest.reporting import (
    build_strategy,
    load_bars_file,
    metrics_as_dict,
    run_backtest_from_bars,
)
from slytrade.cli import app
from slytrade.strategies.baselines import ICTBiasBaselineStrategy, NoTradeStrategy


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


def test_load_bars_file_csv(tmp_path):
    bars = make_bars([100, 101, 102])
    path = tmp_path / "bars.csv"
    bars.to_csv(path, index=False)

    loaded = load_bars_file(path)

    assert len(loaded) == 3
    assert "close" in loaded.columns


def test_build_strategy_factory():
    assert isinstance(build_strategy("no-trade", symbol="XAUUSD", volume=0.1), NoTradeStrategy)
    assert isinstance(build_strategy("ict-bias", symbol="XAUUSD", volume=0.1), ICTBiasBaselineStrategy)


def test_run_backtest_from_bars_no_trade():
    result = run_backtest_from_bars(make_bars([100, 101, 102]), strategy_name="no-trade")
    metrics = metrics_as_dict(result)

    assert metrics["trades"] == 0
    assert result.orders == []


def test_run_backtest_from_bars_ict_bias_computes_features():
    bars = make_bars([1, 2, 3, 6, 3, 2, 1, 4, 7, 8, 9, 8, 7, 6, 5])
    result = run_backtest_from_bars(bars, strategy_name="ict-bias", volume=0.1)

    assert result.metrics.equity_points == len(bars) + 1


def test_run_backtest_cli_outputs_report(tmp_path):
    bars = make_bars([100, 101, 102, 103])
    path = tmp_path / "bars.csv"
    bars.to_csv(path, index=False)
    runner = CliRunner()

    result = runner.invoke(app, ["run-backtest", "--bars-file", str(path), "--strategy", "buy-and-hold", "--volume", "0.1"])

    assert result.exit_code == 0
    assert "Backtest Report" in result.stdout
    assert "Total Return" in result.stdout
