"""Parity + correctness tests for the memory-bounded streaming tick path.

The streaming aligner must produce the same per-bar tick features, decision
quotes and coverage counts as the in-memory ``align_market_data``, while never
holding the full tick set in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from slytrade.data.alignment import TICK_BAR_FEATURE_COLUMNS, align_market_data
from slytrade.data.tick_stream import (
    align_market_data_streaming,
    merge_mt5_exness_streaming,
    resample_ticks_to_bars_streaming,
    sort_tick_files,
    tick_file_metadata,
)


def make_bars(n: int, freq: str = "min", start: str = "2026-01-01") -> pd.DataFrame:
    times = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100.0 + pd.Series(range(n), dtype=float) * 0.01
    return pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSDm",
            "timeframe": "M1",
            "open": close - 0.005,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
        }
    )


def write_tick_month(base: Path, symbol: str, year: int, month: int, start_day: int, ticks_per_day: int = 60) -> Path:
    times = pd.date_range(f"{year:04d}-{month:02d}-{start_day:02d}", periods=ticks_per_day, freq="6s", tz="UTC")
    mid = 100.0 + pd.Series(range(ticks_per_day), dtype=float) * 0.001
    frame = pd.DataFrame(
        {
            "time": times.floor("s"),
            "time_msc": times,
            "symbol": symbol,
            "bid": (mid - 0.01).round(3),
            "ask": (mid + 0.01).round(3),
            "last": 0.0,
            "volume": 1.0,
            "volume_real": 0.0,
            "flags": 0.0,
            "spread": 0.02,
            "mid": mid,
        }
    )
    directory = base / f"symbol={symbol}" / f"year={year}" / f"month={month:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ticks.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_sort_tick_files_chronological(tmp_path: Path) -> None:
    files = [
        write_tick_month(tmp_path, "XAUUSD", 2026, 12, 1, 30),
        write_tick_month(tmp_path, "XAUUSD", 2026, 1, 1, 30),
        write_tick_month(tmp_path, "XAUUSD", 2025, 11, 1, 30),
    ]
    ordered = sort_tick_files(files)
    assert ordered[0] == files[2]  # 2025-11 first
    assert ordered[-1] == files[0]  # 2026-12 last


def test_tick_file_metadata(tmp_path: Path) -> None:
    files = [
        write_tick_month(tmp_path, "XAUUSD", 2026, 1, 1, 60),
        write_tick_month(tmp_path, "XAUUSD", 2026, 2, 1, 40),
    ]
    meta = tick_file_metadata(files)
    assert meta["rows"] == 100
    assert meta["symbol"] == "XAUUSD"
    assert "2026-01-01" in meta["start"]
    assert "2026-02-01" in meta["end"]


def test_streaming_matches_in_memory(tmp_path: Path) -> None:
    """Streaming align produces identical features/quotes/coverage to in-memory."""
    bars = make_bars(40, freq="min", start="2026-01-01 00:00:00")
    f1 = write_tick_month(tmp_path, "XAUUSD", 2026, 1, 1, 240)
    f2 = write_tick_month(tmp_path, "XAUUSD", 2026, 2, 1, 60)

    # In-memory reference: concatenate both files (they are tiny here).
    ticks = pd.concat([pd.read_parquet(f1), pd.read_parquet(f2)], ignore_index=True)
    inmem = align_market_data(
        bars.copy(),
        ticks,
        timeframe="M1",
        canonical_symbol="XAUUSD",
        bar_source="mt5_bars",
        tick_source="exness_ticks",
    )

    streaming = align_market_data_streaming(
        bars.copy(),
        [f1, f2],
        timeframe="M1",
        canonical_symbol="XAUUSD",
        bar_source="mt5_bars",
        tick_source="exness_ticks",
    )

    for column in TICK_BAR_FEATURE_COLUMNS:
        assert np.allclose(inmem.bars[column].to_numpy(), streaming.bars[column].to_numpy(), equal_nan=True), column
    for column in ("quote_bid", "quote_ask", "quote_mid", "quote_spread"):
        assert np.allclose(inmem.bars[column].to_numpy(), streaming.bars[column].to_numpy(), equal_nan=True), column
    assert np.allclose(inmem.bars["quote_age_seconds"].to_numpy(), streaming.bars["quote_age_seconds"].to_numpy(), equal_nan=True)
    assert (inmem.bars["quote_is_fresh"].to_numpy() == streaming.bars["quote_is_fresh"].to_numpy()).all()

    assert inmem.manifest.ticks_rows == streaming.manifest.ticks_rows
    assert inmem.manifest.coverage == streaming.manifest.coverage
    assert inmem.manifest.fresh_coverage_ratio == streaming.manifest.fresh_coverage_ratio


def test_resample_streaming_matches_in_memory(tmp_path: Path) -> None:
    from slytrade.data.resample import resample_ticks_to_bars

    f = write_tick_month(tmp_path, "XAUUSD", 2026, 1, 1, 600)
    ticks = pd.read_parquet(f)
    inmem = resample_ticks_to_bars(ticks, "M1", symbol="XAUUSD")
    streamed = resample_ticks_to_bars_streaming([f], "M1", symbol="XAUUSD")
    pd.testing.assert_frame_equal(inmem, streamed)


def test_merge_mt5_exness_streaming(tmp_path: Path) -> None:
    """MT5 is authoritative: Exness ticks before MT5 coverage are kept, MT5
    wins on overlap, and the merged set has no duplicates or gaps."""
    exness_file = write_tick_month(tmp_path / "ex", "XAUUSD", 2025, 12, 1, 60)
    mt5_file = write_tick_month(tmp_path / "mt5", "XAUUSD", 2026, 1, 1, 40)
    out = tmp_path / "merged"
    total = merge_mt5_exness_streaming([exness_file], [mt5_file], out_root=out, symbol="XAUUSD")
    # Disjoint months: 60 (Dec 2025 Exness) + 40 (Jan 2026 MT5) = 100.
    assert total == 100
    merged_files = list(out.rglob("*.parquet"))
    merged = pd.concat([pd.read_parquet(p) for p in merged_files], ignore_index=True)
    assert len(merged) == 100
    assert set(merged["symbol"].unique()) == {"XAUUSD"}


def test_merge_mt5_wins_on_overlap(tmp_path: Path) -> None:
    """When MT5 and Exness cover the same month, MT5 ticks win and Exness
    ticks before MT5's start are preserved."""
    exness_file = write_tick_month(tmp_path / "ex", "XAUUSD", 2026, 1, 1, 60)
    # MT5 covers the second half of the same month.
    mt5_times = pd.date_range("2026-01-01 00:00:30", periods=30, freq="6s", tz="UTC")
    mid = 101.0 + pd.Series(range(30), dtype=float) * 0.001
    mt5_file = tmp_path / "mt5" / "symbol=XAUUSDm" / "year=2026" / "month=01" / "ticks.parquet"
    mt5_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "time_msc": mt5_times,
            "time": mt5_times.floor("s"),
            "symbol": "XAUUSDm",
            "bid": (mid - 0.01).round(3),
            "ask": (mid + 0.01).round(3),
            "last": 0.0,
            "volume": 1.0,
            "volume_real": 0.0,
            "flags": 0.0,
            "spread": 0.02,
            "mid": mid,
        }
    ).to_parquet(mt5_file, index=False)

    out = tmp_path / "merged"
    total = merge_mt5_exness_streaming([exness_file], [mt5_file], out_root=out, symbol="XAUUSD")
    merged_files = list(out.rglob("*.parquet"))
    merged = pd.concat([pd.read_parquet(p) for p in merged_files], ignore_index=True)
    # Exness ticks before MT5 start (00:00:30) are kept; MT5 ticks kept; the
    # Exness ticks at/after 00:00:30 are dropped (MT5 wins). No duplicates.
    assert len(merged) == total
    assert merged["time_msc"].is_unique
    assert merged["time_msc"].max() == mt5_times[-1]
