from __future__ import annotations

from datetime import UTC, datetime, time

import numpy as np

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


def _hour_label(hour: int) -> str:
    """Session label for a UTC hour-of-day (0..23), matching session_label."""
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 16:
        return "ny_am"
    if 16 <= hour < 21:
        return "ny_pm"
    return "other"


def session_hour_labels(hours: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized session one-hots from UTC hour-of-day values.

    Equivalent to calling ``session_one_hot`` on every row's timestamp, but in
    a single vectorized pass (used by the ICT feature engine's hot loop).
    """
    hours = np.asarray(hours, dtype=int)
    return {
        "session_asia": ((hours >= 0) & (hours < 7)).astype(float),
        "session_london": ((hours >= 7) & (hours < 12)).astype(float),
        "session_ny_am": ((hours >= 12) & (hours < 16)).astype(float),
        "session_ny_pm": ((hours >= 16) & (hours < 21)).astype(float),
        "session_other": ((hours >= 21) | (hours < 0)).astype(float),
    }


def session_one_hot(timestamp: datetime) -> dict[str, float]:
    label = session_label(timestamp)
    return {
        "session_asia": 1.0 if label == "asia" else 0.0,
        "session_london": 1.0 if label == "london" else 0.0,
        "session_ny_am": 1.0 if label == "ny_am" else 0.0,
        "session_ny_pm": 1.0 if label == "ny_pm" else 0.0,
        "session_other": 1.0 if label == "other" else 0.0,
    }

