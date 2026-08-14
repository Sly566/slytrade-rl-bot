"""Red-folder (news) gate.

Professional ICT traders stand aside during high-impact news (the "red folder"
events): liquidity evaporates, spreads widen, and stop hunts become violent. This
gate pauses **new entries** during configured event windows — it never touches
open positions or blocks risk-reducing exits.

Events are loaded from ``configs/news.yaml``. The gate is disabled by default and
stays disabled when the file is missing or ``enabled: false``, so it can never
silently block a deployment. Production operators should load explicit event
windows (from a trusted calendar feed) rather than relying on the optional
recurring approximation of the US Non-Farm Payroll release.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class NewsEvent:
    name: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("news event name cannot be empty")
        if self.end <= self.start:
            raise ValueError(f"news event {self.name!r} must end after it starts")

    def contains(self, now: datetime) -> bool:
        return self.start <= now <= self.end


def _first_friday(year: int, month: int) -> datetime:
    """Return the first Friday of a month at 12:30 UTC (approximate NFP)."""
    day = 1
    while datetime(year, month, day).weekday() != 4:  # 4 == Friday
        day += 1
    return datetime(year, month, day, 12, 30, tzinfo=UTC)


def _recurring_events(year: int) -> tuple[NewsEvent, ...]:
    """Approximate recurring US red-folder events (UTC, no DST adjustment).

    NFP is a fixed 12:30 UTC announcement on the first Friday of each month.
    These are approximations for dry-run convenience only.
    """
    events: list[NewsEvent] = []
    for month in range(1, 13):
        nfp = _first_friday(year, month)
        events.append(NewsEvent(name="NFP", start=nfp, end=nfp + timedelta(minutes=75)))
    return tuple(events)


@dataclass(frozen=True)
class NewsGate:
    """Pauses new entries inside configured (and optional recurring) news windows."""

    enabled: bool = False
    events: tuple[NewsEvent, ...] = ()
    enable_recurring: bool = False
    year: int = 2026
    quiet_before_minutes: int = 15
    quiet_after_minutes: int = 15

    def _effective_events(self) -> tuple[NewsEvent, ...]:
        events = list(self.events)
        if self.enable_recurring:
            events.extend(_recurring_events(self.year))
        padded: list[NewsEvent] = []
        for event in events:
            padded.append(
                NewsEvent(
                    name=event.name,
                    start=event.start - timedelta(minutes=self.quiet_before_minutes),
                    end=event.end + timedelta(minutes=self.quiet_after_minutes),
                )
            )
        return tuple(padded)

    def is_red_folder(self, now: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        else:
            current = current.astimezone(UTC)
        return any(event.contains(current) for event in self._effective_events())

    def reason(self, now: datetime | None = None) -> str | None:
        if not self.enabled:
            return None
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        else:
            current = current.astimezone(UTC)
        for event in self._effective_events():
            if event.contains(current):
                return f"red folder: {event.name} ({event.start.isoformat()} - {event.end.isoformat()})"
        return None


def load_news_gate(path: str | Path | None, *, year: int = 2026) -> NewsGate:
    """Load the news gate from a YAML file. Missing/disabled file -> disabled gate."""
    config_path = Path(path) if path is not None else None
    if config_path is None or not config_path.exists():
        return NewsGate(enabled=False, year=year)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not data.get("enabled", False):
        return NewsGate(enabled=False, year=year)

    events: list[NewsEvent] = []
    for raw in data.get("events", []) or []:
        events.append(
            NewsEvent(
                name=str(raw.get("name", "unnamed")),
                start=_parse_utc(str(raw["start"])),
                end=_parse_utc(str(raw["end"])),
            )
        )
    return NewsGate(
        enabled=True,
        events=tuple(events),
        enable_recurring=bool(data.get("enable_recurring", False)),
        year=int(data.get("year", year)),
        quiet_before_minutes=int(data.get("quiet_before_minutes", 15)),
        quiet_after_minutes=int(data.get("quiet_after_minutes", 15)),
    )
