from __future__ import annotations

from datetime import UTC, datetime

from slytrade.runtime.trading_window import TradingWindow, window_from_settings


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def test_weekday_open_weekend_closed() -> None:
    window = TradingWindow(days=frozenset({"mon", "tue", "wed", "thu", "fri"}), start_utc="00:00", end_utc="23:59")
    assert window.is_open(_dt("2026-08-14T10:00:00+00:00"))  # Friday
    assert not window.is_open(_dt("2026-08-15T10:00:00+00:00"))  # Saturday
    assert "outside trading days" in window.reason(_dt("2026-08-15T10:00:00+00:00"))


def test_intraday_hours_enforced() -> None:
    window = TradingWindow(days=frozenset({"mon", "tue", "wed", "thu", "fri"}), start_utc="07:00", end_utc="16:00")
    assert window.is_open(_dt("2026-08-14T07:00:00+00:00"))
    assert window.is_open(_dt("2026-08-14T15:59:00+00:00"))
    assert not window.is_open(_dt("2026-08-14T06:59:00+00:00"))
    assert not window.is_open(_dt("2026-08-14T16:01:00+00:00"))


def test_overnight_window() -> None:
    window = TradingWindow(days=frozenset({"mon", "tue", "wed", "thu", "fri"}), start_utc="22:00", end_utc="04:00")
    assert window.is_open(_dt("2026-08-14T23:00:00+00:00"))
    assert window.is_open(_dt("2026-08-14T02:00:00+00:00"))
    assert not window.is_open(_dt("2026-08-14T10:00:00+00:00"))


def test_window_from_settings() -> None:
    window = window_from_settings("mon,tue,wed,thu,fri", "07:00", "16:00")
    assert window.days == frozenset({"mon", "tue", "wed", "thu", "fri"})
