"""Economic-calendar feed for the red-folder news gate.

The news gate already pauses new entries around high-impact events; this module
supplies those events from a real feed instead of a hand-edited YAML. Two
sources are supported:

* ``file`` — a JSON/CSV file with ``{name, start_utc, end_utc, impact}`` rows.
* ``url`` — a JSON endpoint returning the same schema (any provider that speaks
  this simple schema; the bot does not vendor a scraper).

The feed is OFF by default and, like the static gate, only ever pauses NEW
entries — never exits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from slytrade.runtime.news_gate import NewsEvent, NewsGate


@dataclass(frozen=True)
class CalendarEntry:
    name: str
    start_utc: str
    end_utc: str
    impact: str = "high"  # low | medium | high
    currency: str = "USD"

    def to_event(self) -> NewsEvent:
        start = _parse_utc(self.start_utc)
        end = _parse_utc(self.end_utc)
        return NewsEvent(name=self.name, start=start, end=end)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _rows_from_payload(payload: Any) -> list[CalendarEntry]:
    if isinstance(payload, dict):
        payload = payload.get("events", payload.get("data", []))
    entries: list[CalendarEntry] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        entries.append(
            CalendarEntry(
                name=str(row.get("name", "event")),
                start_utc=str(row.get("start_utc") or row.get("start")),
                end_utc=str(row.get("end_utc") or row.get("end")),
                impact=str(row.get("impact", "high")),
                currency=str(row.get("currency", "USD")),
            )
        )
    return entries


def load_calendar_entries(
    *,
    path: str | None = None,
    url: str | None = None,
    min_impact: str = "high",
) -> list[CalendarEntry]:
    """Load calendar entries from a file or URL, filtered by minimum impact."""
    if path is not None:
        source = Path(path)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            frame = pd.read_csv(source)
            payload = frame.to_dict(orient="records")
    elif url is not None:
        import urllib.request

        with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310 - operator-configured URL
            payload = json.loads(response.read().decode("utf-8"))
    else:
        return []

    rank = {"low": 0, "medium": 1, "high": 2}
    threshold = rank.get(min_impact.lower(), 2)
    return [entry for entry in _rows_from_payload(payload) if rank.get(entry.impact.lower(), 0) >= threshold]


def calendar_gate(
    *,
    path: str | None = None,
    url: str | None = None,
    min_impact: str = "high",
    quiet_before_minutes: int = 15,
    quiet_after_minutes: int = 15,
) -> NewsGate:
    """Build a NewsGate from a calendar feed (disabled when no source is given)."""
    entries = load_calendar_entries(path=path, url=url, min_impact=min_impact)
    if not entries:
        return NewsGate(enabled=False)
    events = tuple(entry.to_event() for entry in entries)
    return NewsGate(
        enabled=True,
        events=events,
        quiet_before_minutes=quiet_before_minutes,
        quiet_after_minutes=quiet_after_minutes,
    )


def build_news_gate_from_settings(settings) -> NewsGate:
    """Build the runtime news gate from settings + configs/news.yaml.

    Precedence when ``news_enabled`` is set: settings calendar URL/path →
    news.yaml ``calendar_url``/``calendar_path`` → static news.yaml events →
    recurring approximations. Returns a disabled gate unless enabled.
    """
    from datetime import UTC, datetime
    from pathlib import Path as _Path

    import yaml as _yaml

    from slytrade.runtime.news_gate import NewsGate, load_news_gate

    if not getattr(settings, "news_enabled", False):
        return NewsGate(enabled=False)

    news_cfg: dict = {}
    news_path = _Path(getattr(settings, "news_config_file", "configs/news.yaml"))
    if news_path.exists():
        try:
            news_cfg = _yaml.safe_load(news_path.read_text(encoding="utf-8")) or {}
        except Exception:  # pragma: no cover - malformed operator file
            news_cfg = {}

    # Calendar feed source: env/settings wins, then news.yaml.
    calendar_path = getattr(settings, "calendar_path", "") or str(news_cfg.get("calendar_path", ""))
    calendar_url = getattr(settings, "calendar_url", "") or str(news_cfg.get("calendar_url", ""))
    min_impact = getattr(settings, "news_min_impact", "high") or str(news_cfg.get("min_impact", "high"))

    if calendar_path or calendar_url:
        gate = calendar_gate(
            path=calendar_path or None,
            url=calendar_url or None,
            min_impact=min_impact,
            quiet_before_minutes=int(news_cfg.get("quiet_before_minutes", 15)),
            quiet_after_minutes=int(news_cfg.get("quiet_after_minutes", 15)),
        )
        if gate.enabled:
            return gate

    return load_news_gate(news_path, year=datetime.now(UTC).year)
