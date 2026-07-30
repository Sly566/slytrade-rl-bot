from __future__ import annotations

from datetime import UTC, datetime, time

SESSION_COLUMNS = [
    "session_asia",
    "session_london",
    "session_ny_am",
    "session_ny_pm",
    "session_other",
]


def _utc_time(value: datetime) -> time:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).time()


def _in_range(current: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def session_label(timestamp: datetime) -> str:
    """Return a simple UTC trading-session label.

    The labels are intentionally broad. Later phases can make this broker- and
    symbol-aware, but these defaults are a stable causal starting point.
    """
    t = _utc_time(timestamp)
    if _in_range(t, time(0, 0), time(7, 0)):
        return "asia"
    if _in_range(t, time(7, 0), time(12, 0)):
        return "london"
    if _in_range(t, time(12, 0), time(16, 0)):
        return "ny_am"
    if _in_range(t, time(16, 0), time(21, 0)):
        return "ny_pm"
    return "other"


def session_one_hot(timestamp: datetime) -> dict[str, float]:
    label = session_label(timestamp)
    return {
        "session_asia": 1.0 if label == "asia" else 0.0,
        "session_london": 1.0 if label == "london" else 0.0,
        "session_ny_am": 1.0 if label == "ny_am" else 0.0,
        "session_ny_pm": 1.0 if label == "ny_pm" else 0.0,
        "session_other": 1.0 if label == "other" else 0.0,
    }
