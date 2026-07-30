from datetime import UTC, datetime

import pandas as pd

from slytrade.data.schemas import normalize_bar_frame, normalize_tick_frame
from slytrade.data.validators import validate_bar_frame, validate_tick_frame


def test_tick_normalization_adds_spread_and_mid():
    raw = [
        {"time": 1767225600, "time_msc": 1767225600000, "bid": 2400.0, "ask": 2400.2, "last": 0.0, "volume": 1},
    ]

    frame = normalize_tick_frame(raw, "XAUUSD")

    assert list(frame["symbol"].unique()) == ["XAUUSD"]
    assert round(float(frame.loc[0, "spread"]), 2) == 0.2
    assert round(float(frame.loc[0, "mid"]), 2) == 2400.1


def test_tick_validation_drops_duplicates_and_bad_prices():
    raw = [
        {"time": 1767225600, "time_msc": 1767225600000, "bid": 2400.0, "ask": 2400.2, "last": 0.0},
        {"time": 1767225600, "time_msc": 1767225600000, "bid": 2400.0, "ask": 2400.2, "last": 0.0},
        {"time": 1767225601, "time_msc": 1767225601000, "bid": -1.0, "ask": 2400.3, "last": 0.0},
    ]
    frame = normalize_tick_frame(raw, "XAUUSD")

    clean, report = validate_tick_frame(frame)

    assert len(clean) == 1
    assert report.rows_before == 3
    assert report.rows_after == 1
    assert any(issue.code == "duplicate_ticks" for issue in report.issues)
    assert any(issue.code == "bad_tick_prices" for issue in report.issues)


def test_bar_normalization_and_validation():
    raw = [
        {"time": 1767225600, "open": 2400.0, "high": 2401.0, "low": 2399.0, "close": 2400.5, "tick_volume": 10},
        {"time": 1767225660, "open": 2400.5, "high": 2400.2, "low": 2399.5, "close": 2400.7, "tick_volume": 12},
    ]
    frame = normalize_bar_frame(raw, "XAUUSD", "M1")

    clean, report = validate_bar_frame(frame)

    assert len(clean) == 1
    assert report.rows_before == 2
    assert any(issue.code == "invalid_ohlc_range" for issue in report.issues)


def test_datetime_input_supported_for_ticks():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    frame = normalize_tick_frame(pd.DataFrame([{"time": now, "bid": 1.0, "ask": 1.1}]), "EURUSD")

    assert str(frame.loc[0, "time"].tz) == "UTC"
