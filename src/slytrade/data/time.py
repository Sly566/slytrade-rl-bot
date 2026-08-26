from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Literal

ChunkSize = Literal["day", "week", "month"]

_LOOKBACK_RE = re.compile(r"^(?P<count>\d+)(?P<unit>d|day|days|w|week|weeks|m|mo|month|months|y|year|years)$", re.IGNORECASE)


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_utc_datetime(value: str) -> datetime:
    """Parse an ISO date/datetime string as UTC.

    Examples accepted:
    - 2026-01-01
    - 2026-01-01T13:30:00
    - 2026-01-01T13:30:00+00:00
    """
    parsed = datetime.fromisoformat(value)
    return ensure_utc(parsed)


def parse_lookback_duration(value: str) -> timedelta:
    """Parse lookback duration strings used by CLI data collection.

    Supported examples:
    - 1d, 7d
    - 1w, 4w
    - 1m, 6m  (calendar approximation: 30 days each)
    - 1y, 2y  (calendar approximation: 365 days each)
    """
    normalized = value.strip().lower().replace(" ", "")
    match = _LOOKBACK_RE.match(normalized)
    if not match:
        raise ValueError("lookback must look like 1d, 1w, 1m, 6m, 1y or 2y")
    count = int(match.group("count"))
    unit = match.group("unit")
    if count <= 0:
        raise ValueError("lookback count must be positive")
    if unit in {"d", "day", "days"}:
        return timedelta(days=count)
    if unit in {"w", "week", "weeks"}:
        return timedelta(days=7 * count)
    if unit in {"m", "mo", "month", "months"}:
        return timedelta(days=30 * count)
    if unit in {"y", "year", "years"}:
        return timedelta(days=365 * count)
    raise ValueError(f"unsupported lookback unit: {unit}")


def date_range_from_lookback(lookback: str, *, end: datetime | None = None) -> tuple[datetime, datetime]:
    end_dt = ensure_utc(end) if end is not None else utc_now()
    start_dt = end_dt - parse_lookback_duration(lookback)
    return start_dt, end_dt


# --------------------------------------------------------------------------- #
# Timeframe helpers (used by mtf_align.py, per_tf.py)
# --------------------------------------------------------------------------- #

_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "M1":  timedelta(minutes=1),
    "M2":  timedelta(minutes=2),
    "M3":  timedelta(minutes=3),
    "M4":  timedelta(minutes=4),
    "M5":  timedelta(minutes=5),
    "M6":  timedelta(minutes=6),
    "M10": timedelta(minutes=10),
    "M12": timedelta(minutes=12),
    "M15": timedelta(minutes=15),
    "M20": timedelta(minutes=20),
    "M30": timedelta(minutes=30),
    "H1":  timedelta(hours=1),
    "H2":  timedelta(hours=2),
    "H3":  timedelta(hours=3),
    "H4":  timedelta(hours=4),
    "H6":  timedelta(hours=6),
    "H8":  timedelta(hours=8),
    "H12": timedelta(hours=12),
    "D1":  timedelta(days=1),
    "W1":  timedelta(days=7),
    "MN1": timedelta(days=30),
}


def timeframe_timedelta(timeframe: str) -> timedelta:
    """Return the timedelta duration for an MT5-style timeframe string."""
    tf = timeframe.upper().strip()
    try:
        return _TIMEFRAME_DELTAS[tf]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}") from exc


def timeframe_minutes(timeframe: str) -> int:
    return int(timeframe_timedelta(timeframe).total_seconds() // 60)


def iter_months(start: datetime, end: datetime):
    """Yield (year, month) tuples covering [start, end)."""
    start = ensure_utc(start); end = ensure_utc(end)
    y, m = start.year, start.month
    while True:
        yield (y, m)
        if m == 12:
            y += 1; m = 1
        else:
            m += 1
        if (y, m) > (end.year, end.month):
            break


def _common_dt_unit(*series):
    """Cast datetime64 Series to a common finest unit (pandas-3 merge_asof safe)."""
    import pandas as pd
    unit_order = {"ns": 0, "us": 1, "ms": 2, "s": 3}
    finest = None
    casted = []
    for s in series:
        s2 = pd.to_datetime(s, utc=True, errors="coerce")
        try:
            u = s2.dt.unit
        except AttributeError:
            u = "ns"
        if finest is None or unit_order.get(u, 99) < unit_order.get(finest, 99):
            finest = u
        casted.append(s2)
    if finest is None:
        finest = "ns"
    out = []
    for s in casted:
        try:
            out.append(pd.Series(s.array.as_unit(finest), index=s.index, name=s.name))
        except AttributeError:
            out.append(s)
    return tuple(out)


def iter_time_chunks(start: datetime, end: datetime, chunk_size: ChunkSize) -> list[tuple[datetime, datetime]]:
    """Split a date range into deterministic UTC chunks."""
    start = ensure_utc(start)
    end = ensure_utc(end)
    if start >= end:
        raise ValueError("start must be earlier than end")

    chunks: list[tuple[datetime, datetime]] = []
    current = start
    while current < end:
        if chunk_size == "day":
            nxt = min(current + timedelta(days=1), end)
        elif chunk_size == "week":
            nxt = min(current + timedelta(days=7), end)
        elif chunk_size == "month":
            year = current.year + (1 if current.month == 12 else 0)
            month = 1 if current.month == 12 else current.month + 1
            nxt = min(current.replace(year=year, month=month, day=1), end)
        else:
            raise ValueError(f"Unsupported chunk size: {chunk_size}")
        chunks.append((current, nxt))
        current = nxt
    return chunks
