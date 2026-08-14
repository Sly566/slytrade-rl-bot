from __future__ import annotations

import pandas as pd

from slytrade.data.resample import resample_ticks_to_bars
from slytrade.data.schemas import BAR_COLUMNS


def make_ticks(minutes: int = 60, freq: str = "6s") -> pd.DataFrame:
    times = pd.date_range("2026-08-14T10:00:00", periods=int(minutes * 60 / int(freq.rstrip("s"))), freq=freq, tz="UTC")
    mid = 100.0 + pd.Series(range(len(times)), dtype=float) * 0.001
    return pd.DataFrame({"time_msc": times, "symbol": "XAUUSD", "bid": (mid - 0.01).round(3), "ask": (mid + 0.01).round(3)})


def test_resample_m1() -> None:
    bars = resample_ticks_to_bars(make_ticks(minutes=60), "M1")
    assert list(bars.columns) == BAR_COLUMNS
    assert len(bars) == 60
    # Timestamped at bar open, minute-grid aligned.
    assert bars.iloc[0]["time"] == pd.Timestamp("2026-08-14T10:00:00", tz="UTC")
    assert bars.iloc[1]["time"] == pd.Timestamp("2026-08-14T10:01:00", tz="UTC")
    # OHLC invariants.
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["low"] <= bars["open"]).all()


def test_resample_m5_and_h1() -> None:
    ticks = make_ticks(minutes=120)
    assert len(resample_ticks_to_bars(ticks, "M5")) == 24
    assert len(resample_ticks_to_bars(ticks, "H1")) == 2
    assert len(resample_ticks_to_bars(ticks, "M15")) == 8


def test_spread_and_volume_captured() -> None:
    bars = resample_ticks_to_bars(make_ticks(minutes=10), "M1")
    assert bars["tick_volume"].sum() == 10 * 10  # 10 ticks per minute bar
    assert (bars["spread"] > 0).all()


def test_unsupported_timeframe_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        resample_ticks_to_bars(make_ticks(minutes=5), "ZZ")


def test_empty_ticks_returns_empty_bars() -> None:
    empty = pd.DataFrame(columns=["time_msc", "bid", "ask", "symbol"])
    bars = resample_ticks_to_bars(empty, "M1")
    assert list(bars.columns) == BAR_COLUMNS
    assert bars.empty


def test_symbol_inferred_from_frame() -> None:
    bars = resample_ticks_to_bars(make_ticks(minutes=5), "M1")
    assert set(bars["symbol"].unique()) == {"XAUUSD"}
