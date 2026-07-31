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
