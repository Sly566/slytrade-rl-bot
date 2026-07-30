import pandas as pd
from typer.testing import CliRunner

from slytrade.cli import app
from slytrade.data.sample_generator import generate_sample_bars, generate_sample_ticks, write_sample_frame
from slytrade.data.time import parse_utc_datetime


def test_generate_sample_bars_is_deterministic():
    start = parse_utc_datetime("2026-01-01")
    first = generate_sample_bars(start=start, periods=10, seed=123)
    second = generate_sample_bars(start=start, periods=10, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 10
    assert {"time", "symbol", "timeframe", "open", "high", "low", "close", "tick_volume", "spread"}.issubset(first.columns)
    assert (first["high"] >= first[["open", "close"]].max(axis=1)).all()
    assert (first["low"] <= first[["open", "close"]].min(axis=1)).all()


def test_generate_sample_ticks_has_bid_ask_spread_and_mid():
    start = parse_utc_datetime("2026-01-01")
    ticks = generate_sample_ticks(start=start, periods=20, seed=123)

    assert len(ticks) == 20
    assert (ticks["ask"] >= ticks["bid"]).all()
    assert (ticks["spread"] >= 0).all()
    assert ((ticks["bid"] + ticks["ask"]) / 2.0).round(10).equals(ticks["mid"].round(10))


def test_write_sample_frame_csv(tmp_path):
    frame = generate_sample_bars(start=parse_utc_datetime("2026-01-01"), periods=5)
    path = write_sample_frame(frame, tmp_path / "bars.csv")

    assert path.exists()
    assert len(pd.read_csv(path)) == 5


def test_generate_sample_bars_cli(tmp_path):
    output = tmp_path / "sample_bars.csv"
    runner = CliRunner()

    result = runner.invoke(app, ["generate-sample-bars", "--output-file", str(output), "--periods", "12"])

    assert result.exit_code == 0
    assert output.exists()
    assert len(pd.read_csv(output)) == 12


def test_generate_sample_ticks_cli(tmp_path):
    output = tmp_path / "sample_ticks.csv"
    runner = CliRunner()

    result = runner.invoke(app, ["generate-sample-ticks", "--output-file", str(output), "--periods", "12"])

    assert result.exit_code == 0
    assert output.exists()
    assert len(pd.read_csv(output)) == 12


def test_generated_bars_can_run_backtest_cli(tmp_path):
    output = tmp_path / "sample_bars.csv"
    runner = CliRunner()
    generated = runner.invoke(app, ["generate-sample-bars", "--output-file", str(output), "--periods", "50"])
    assert generated.exit_code == 0

    backtest = runner.invoke(app, ["compare-baselines", "--bars-file", str(output)])

    assert backtest.exit_code == 0
    assert "Baseline Comparison" in backtest.stdout
