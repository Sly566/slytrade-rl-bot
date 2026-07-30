from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

ChunkSize = Literal["day", "week", "month"]


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_utc_datetime(value: str) -> datetime:
    """Parse an ISO date/datetime string as UTC.

    Examples accepted:
    - 2026-01-01
    - 2026-01-01T13:30:00
    - 2026-01-01T13:30:00+00:00
    """
    parsed = datetime.fromisoformat(value)
    return ensure_utc(parsed)


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
