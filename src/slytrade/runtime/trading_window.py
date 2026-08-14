"""Session / trading-window enforcement.

Professional ICT traders only take setups inside institutional kill zones and
avoid thin, illiquid windows (weekends, rollover). This module enforces the
*structural* trading window (weekdays + UTC hours) as a hard gate so the loop
never opens new risk outside approved hours. Kill-zone *preferences* remain a
strategy-level signal, not a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time


@dataclass(frozen=True)
class TradingWindow:
    days: frozenset[str] = frozenset({"mon", "tue", "wed", "thu", "fri"})
    start_utc: str = "00:00"
    end_utc: str = "23:59"

    def is_open(self, now: datetime | None = None) -> bool:
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        else:
            current = current.astimezone(UTC)
        day = current.strftime("%a").lower()
        if day not in self.days:
            return False
        start = time.fromisoformat(self.start_utc)
        end = time.fromisoformat(self.end_utc)
        current_time = current.time().replace(tzinfo=None)
        if start <= end:
            return start <= current_time <= end
        # Overnight window (e.g. 22:00 -> 04:00).
        return current_time >= start or current_time <= end

    def reason(self, now: datetime | None = None) -> str:
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        else:
            current = current.astimezone(UTC)
        day = current.strftime("%a").lower()
        if day not in self.days:
            return f"outside trading days ({day})"
        return f"outside trading hours {self.start_utc}-{self.end_utc} UTC"


def window_from_settings(days: str, start_utc: str, end_utc: str) -> TradingWindow:
    return TradingWindow(
        days=frozenset(day.strip().lower() for day in days.split(",") if day.strip()),
        start_utc=start_utc,
        end_utc=end_utc,
    )
