"""Timeframe constants, helpers and UTC-safe datetime utilities for Layer 1."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Iterable

# Canonical minutes-per-bar for every timeframe the project uses.
TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 1440 * 7,
}


def timeframe_minutes(tf: str) -> int:
    """Return the number of minutes in a single bar of `tf`."""
    key = tf.upper()
    try:
        return TIMEFRAME_MINUTES[key]
    except KeyError as e:
        raise ValueError(f"Unsupported timeframe: {tf}") from e


def timeframe_timedelta(tf: str) -> timedelta:
    """Return the duration of one bar of `tf` as a timedelta."""
    return timedelta(minutes=timeframe_minutes(tf))


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime (localize naive datetimes as UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)


def floor_to_timeframe(ts: datetime, tf: str) -> datetime:
    """Floor a UTC datetime to the most recent bar-open of timeframe `tf`.

    Bar grid starts at midnight UTC for intraday/D1 TFs and Monday midnight
    UTC for W1. This matches MT5's convention where bars are timestamped at
    their open time.
    """
    ts = ensure_utc(ts)
    minutes = timeframe_minutes(tf)
    if tf.upper() == "W1":
        # Monday 00:00 UTC of the week containing ts.
        monday = ts - timedelta(days=ts.weekday())
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)
    if minutes >= 1440:
        # D1 and longer: midnight UTC.
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    # Intraday: floor to N-minute grid from midnight.
    midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = int((ts - midnight).total_seconds() // 60)
    floored = (elapsed // minutes) * minutes
    return midnight + timedelta(minutes=floored)


def add_decision_time(bars, tf: str | None = None):
    """Add a `decision_time` column = bar open + 1 bar duration.

    Causal rule: a bar's OHLC is only known AFTER the bar closes, so the
    earliest moment a decision can use that bar is bar_open + bar_length.
    Accepts and returns a pandas DataFrame; imported lazily so the rest of
    this module stays importable without pandas.
    """
    import pandas as pd  # noqa: PLC0415 – lazy to keep module import light.

    if "time" not in bars.columns:
        raise ValueError("bars must contain a 'time' column")
    out = bars.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    tf_key = (tf or (out["timeframe"].iloc[0] if "timeframe" in out.columns and not out.empty else None))
    if tf_key is None:
        raise ValueError("timeframe must be provided when bars lack a timeframe column")
    out["decision_time"] = out["time"] + timeframe_timedelta(str(tf_key).upper())
    return out


def _common_dt_unit(series_a, series_b):
    """Cast two datetime Series to a common resolution (ns) for merge_asof.

    Pandas 3 stores datetimes as ``us`` by default; ``merge_asof`` on two
    series with different units raises, so we normalise both sides to ns.
    """
    import pandas as pd  # noqa: PLC0415

    a = pd.to_datetime(series_a, utc=True, errors="coerce")
    b = pd.to_datetime(series_b, utc=True, errors="coerce")
    return a.array.as_unit("ns"), b.array.as_unit("ns")


def iter_days(start: date, end: date) -> Iterable[date]:
    """Yield each date in ``[start, end)``."""
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=1)


def iter_months(start: date, end: date) -> Iterable[tuple[date, date]]:
    """Yield ``(month_start, next_month_start)`` pairs covering [start, end)."""
    cur = date(start.year, start.month, 1)
    while cur < end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        yield cur, nxt
        cur = nxt
