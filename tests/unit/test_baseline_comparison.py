import numpy as np
import pandas as pd
from typer.testing import CliRunner

from slytrade.backtest.reporting import (
    VALID_STRATEGIES,
    compare_baselines_from_bars,
    comparison_as_frame,
    render_baseline_comparison,
)
from slytrade.cli import app


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


def test_compare_baselines_runs_all_strategies():
    rows = compare_baselines_from_bars(make_bars([100, 101, 102, 103, 104, 103, 102, 105]), volume=0.1)

    assert {row.strategy for row in rows} == set(VALID_STRATEGIES)
    assert rows == sorted(rows, key=lambda row: row.final_equity, reverse=True)


def test_comparison_as_frame_has_expected_columns():
    rows = compare_baselines_from_bars(make_bars([100, 101, 102, 103]), strategies=("no-trade", "buy-and-hold"))
    frame = comparison_as_frame(rows)

    assert list(frame["strategy"]) == [row.strategy for row in rows]
    assert "final_equity" in frame.columns
    assert "total_return" in frame.columns


def test_render_baseline_comparison_outputs_table(capsys):
    rows = compare_baselines_from_bars(make_bars([100, 101, 102, 103]), strategies=("no-trade",))

    render_baseline_comparison(rows)
    captured = capsys.readouterr()

    assert "Baseline Comparison" in captured.out
    assert "no-trade" in captured.out


def test_compare_baselines_cli_outputs_report_and_csv(tmp_path):
    bars = make_bars([100, 101, 102, 103, 104, 105])
    bars_path = tmp_path / "bars.csv"
    out_path = tmp_path / "comparison.csv"
    bars.to_csv(bars_path, index=False)
    runner = CliRunner()

    result = runner.invoke(app, ["compare-baselines", "--bars-file", str(bars_path), "--output-csv", str(out_path)])

    assert result.exit_code == 0
    assert "Baseline Comparison" in result.stdout
    assert out_path.exists()
    saved = pd.read_csv(out_path)
    assert set(saved["strategy"]) == set(VALID_STRATEGIES)
