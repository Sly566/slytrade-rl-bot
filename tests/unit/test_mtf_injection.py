"""Tests for the MTF injection and clean/reset behaviour in the task layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from slytrade import tasks


def make_mtf_bars(symbol: str, tf: str, n: int, start: str) -> pd.DataFrame:
    times = pd.date_range(start, periods=n, freq="min", tz="UTC")
    close = 100.0 + pd.Series(range(n), dtype=float) * 0.01
    return pd.DataFrame(
        {
            "time": times,
            "symbol": symbol,
            "timeframe": tf,
            "open": close - 0.005,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
        }
    )


def write_bars(root: Path, symbol: str, tf: str, frame: pd.DataFrame) -> None:
    directory = root / "mt5_bars" / f"symbol={symbol}" / f"timeframe={tf}" / "y=2026"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "bars.parquet", index=False)


def test_inject_mtf_features_adds_mtf_columns(tmp_path: Path, monkeypatch) -> None:
    from slytrade.data.alignment import AlignedDataset, DatasetManifest

    symbol = "XAUUSDm"
    # M1 aligned bars (the execution frame).
    m1 = make_mtf_bars(symbol, "M1", 120, "2026-01-01 00:00:00")
    m1 = m1.rename(columns={"symbol": "symbol"})
    m1["decision_time"] = m1["time"] + pd.Timedelta(minutes=1)
    m1["quote_is_fresh"] = True
    # Higher timeframes collected from MT5.
    write_bars(tmp_path, symbol, "M5", make_mtf_bars(symbol, "M5", 40, "2026-01-01 00:00:00"))
    write_bars(tmp_path, symbol, "H1", make_mtf_bars(symbol, "H1", 20, "2026-01-01 00:00:00"))

    manifest = DatasetManifest(
        canonical_symbol="XAUUSD",
        bar_symbol=symbol,
        tick_symbol="XAUUSD",
        bar_source="mt5_bars",
        tick_source="exness_ticks",
        timeframe="M1",
        bars_rows=len(m1),
        ticks_rows=0,
        bars_start=str(m1["time"].min()),
        bars_end=str(m1["time"].max()),
        decision_start=str(m1["decision_time"].min()),
        decision_end=str(m1["decision_time"].max()),
        ticks_start="",
        ticks_end="",
        coverage={},
    )
    dataset = AlignedDataset(bars=m1, ticks=pd.DataFrame(), manifest=manifest)

    frames = {
        "M5": make_mtf_bars(symbol, "M5", 40, "2026-01-01 00:00:00"),
        "H1": make_mtf_bars(symbol, "H1", 20, "2026-01-01 00:00:00"),
    }

    def load(sym, tf, root):
        if tf not in frames:
            raise FileNotFoundError(tf)
        return frames[tf]

    monkeypatch.setattr(tasks, "load_collected_bars", load)

    result = tasks._inject_mtf_features(dataset, "XAUUSD", "M1", tmp_path)
    bars = result.bars
    assert "mtf_bias" in bars.columns
    assert "mtf_confluence_score" in bars.columns
    assert any("htf_m5_" in column for column in bars.columns)
    assert any("htf_h1_" in column for column in bars.columns)


def test_inject_mtf_returns_unchanged_when_no_htf(tmp_path: Path, monkeypatch) -> None:
    from slytrade.data.alignment import AlignedDataset, DatasetManifest

    m1 = make_mtf_bars("XAUUSDm", "M1", 60, "2026-01-01 00:00:00")
    m1["decision_time"] = m1["time"] + pd.Timedelta(minutes=1)
    m1["quote_is_fresh"] = True
    manifest = DatasetManifest(
        canonical_symbol="XAUUSD", bar_symbol="XAUUSDm", tick_symbol="XAUUSD",
        bar_source="mt5_bars", tick_source="exness_ticks", timeframe="M1",
        bars_rows=len(m1), ticks_rows=0, bars_start="", bars_end="",
        decision_start="", decision_end="", ticks_start="", ticks_end="", coverage={},
    )
    dataset = AlignedDataset(bars=m1, ticks=pd.DataFrame(), manifest=manifest)

    def no_bars(sym, tf, root):
        raise FileNotFoundError

    monkeypatch.setattr(tasks, "load_collected_bars", no_bars)
    result = tasks._inject_mtf_features(dataset, "XAUUSD", "M1", tmp_path)
    assert "mtf_bias" not in result.bars.columns


def test_clean_all_keeps_dirs_and_wipes_contents(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Create a stale file inside data/raw (the thing the user wants gone).
    stale_dir = tmp_path / "data" / "raw" / "symbol=XAUUSDm"
    stale_dir.mkdir(parents=True)
    (stale_dir / "old.parquet").write_text("stale", encoding="utf-8")
    (tmp_path / "models" / "registry.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "registry.jsonl").write_text("old", encoding="utf-8")

    result = tasks.clean_all()
    assert result.ok, result.message
    # Directories kept (ownership preserved), contents removed.
    assert (tmp_path / "data" / "raw").is_dir()
    assert not (tmp_path / "data" / "raw" / "symbol=XAUUSDm").exists()
    assert (tmp_path / "models").is_dir()
    assert not (tmp_path / "models" / "registry.jsonl").exists()
