import numpy as np
import pandas as pd
from typer.testing import CliRunner

from slytrade.cli import app
from slytrade.data.diagnostics import inspect_bars, inspect_tick_bar_coverage, inspect_ticks
from slytrade.data.sample_generator import generate_sample_bars, generate_sample_ticks
from slytrade.data.time import parse_utc_datetime
from slytrade.data.timeframes import add_decision_time, decision_time_for_bar, timeframe_duration


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
            "tick_volume": [10, 11, 12],
            "spread": [10, 10, 10],
            "real_volume": [0, 0, 0],
        }
    )


def make_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC").floor("s"),
            "time_msc": pd.date_range("2026-01-01", periods=3, freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "bid": [100.0, 101.0, 102.0],
            "ask": [100.2, 101.2, 102.2],
            "last": np.zeros(3),
            "volume": np.ones(3),
            "volume_real": np.zeros(3),
            "flags": np.zeros(3),
            "spread": [0.2, 0.2, 0.2],
            "mid": [100.1, 101.1, 102.1],
        }
    )


def test_timeframe_duration_and_decision_time():
    assert timeframe_duration("M1").total_seconds() == 60
    bars = add_decision_time(make_bars())

    assert str(bars.loc[0, "decision_time"]) == "2026-01-01 00:01:00+00:00"
    assert str(decision_time_for_bar(bars.iloc[0])) == "2026-01-01 00:01:00+00:00"


def test_data_diagnostics():
    bars_diag = inspect_bars(make_bars())
    ticks_diag = inspect_ticks(make_ticks())
    coverage = inspect_tick_bar_coverage(make_bars(), make_ticks())

    assert bars_diag.rows == 3
    assert bars_diag.start_decision_time == "2026-01-01 00:01:00+00:00"
    assert ticks_diag.rows == 3
    assert ticks_diag.spread_mean == 0.20000000000000284
    assert coverage.bars == 3
    assert coverage.bars_with_tick_before_decision >= 2


def test_inspect_data_cli(tmp_path):
    start = parse_utc_datetime("2026-01-01")
    bars_path = tmp_path / "bars.csv"
    ticks_path = tmp_path / "ticks.csv"
    generate_sample_bars(start=start, periods=10).to_csv(bars_path, index=False)
    generate_sample_ticks(start=start, periods=20).to_csv(ticks_path, index=False)
    runner = CliRunner()

    result = runner.invoke(app, ["inspect-data", "--bars-file", str(bars_path), "--ticks-file", str(ticks_path)])

    assert result.exit_code == 0
    assert "Data Diagnostics" in result.stdout
    assert "decision" in result.stdout
