import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from slytrade.cli import app
from slytrade.data.alignment import align_market_data, infer_canonical_symbol, load_manifest, save_aligned_dataset


def make_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-01", periods=3, freq="min", tz="UTC"),
            "symbol": "XAUUSDm",
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


def make_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-01T00:01:00Z", periods=3, freq="min").floor("s"),
            "time_msc": pd.date_range("2026-07-01T00:01:00Z", periods=3, freq="min"),
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


def test_infer_canonical_symbol_from_aliases():
    assert infer_canonical_symbol("XAUUSDm", "XAUUSD") == "XAUUSD"


def test_align_market_data_normalizes_symbols_and_manifest():
    dataset = align_market_data(make_bars(), make_ticks(), timeframe="M1")

    assert dataset.bars["symbol"].unique().tolist() == ["XAUUSD"]
    assert dataset.ticks["symbol"].unique().tolist() == ["XAUUSD"]
    assert dataset.manifest.canonical_symbol == "XAUUSD"
    assert dataset.manifest.bar_symbol == "XAUUSDm"
    assert dataset.manifest.tick_symbol == "XAUUSD"
    assert dataset.manifest.coverage["bars_with_fresh_tick_before_decision"] == 3


def test_save_and_load_aligned_dataset(tmp_path):
    dataset = align_market_data(make_bars(), make_ticks(), timeframe="M1")
    manifest = save_aligned_dataset(dataset, tmp_path / "aligned")
    loaded = load_manifest(tmp_path / "aligned" / "manifest.json")

    assert (tmp_path / "aligned" / "manifest.json").exists()
    assert manifest.files["bars"]
    assert manifest.files["ticks"]
    assert loaded.canonical_symbol == "XAUUSD"


def test_align_dataset_cli(tmp_path):
    bars_path = tmp_path / "bars.csv"
    ticks_path = tmp_path / "ticks.csv"
    out_dir = tmp_path / "dataset"
    make_bars().to_csv(bars_path, index=False)
    make_ticks().to_csv(ticks_path, index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "align-dataset",
            "--bars-file",
            str(bars_path),
            "--ticks-file",
            str(ticks_path),
            "--output-dir",
            str(out_dir),
            "--timeframe",
            "M1",
        ],
    )

    assert result.exit_code == 0
    assert "Aligned Dataset Manifest" in result.stdout
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["canonical_symbol"] == "XAUUSD"


def test_align_market_data_can_drop_stale_bars():
    bars = make_bars()
    ticks = make_ticks().iloc[:1].copy()

    dataset = align_market_data(
        bars,
        ticks,
        timeframe="M1",
        require_fresh_quotes=True,
        min_fresh_coverage=0.95,
    )

    assert len(dataset.bars) == 1
    assert dataset.manifest.source_bars_rows == 3
    assert dataset.manifest.aligned_bars_rows == 1
    assert dataset.manifest.dropped_stale_bars == 2
    assert dataset.manifest.require_fresh_quotes is True
    assert dataset.manifest.quality_status == "WARN"
