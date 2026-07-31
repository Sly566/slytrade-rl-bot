from datetime import UTC, datetime

from slytrade.data.time import date_range_from_lookback, parse_lookback_duration


def test_parse_lookback_duration():
    assert parse_lookback_duration("1d").days == 1
    assert parse_lookback_duration("2w").days == 14
    assert parse_lookback_duration("1m").days == 30
    assert parse_lookback_duration("2y").days == 730


def test_date_range_from_lookback_with_explicit_end():
    end = datetime(2026, 7, 31, tzinfo=UTC)
    start, resolved_end = date_range_from_lookback("1y", end=end)

    assert resolved_end == end
    assert (resolved_end - start).days == 365
