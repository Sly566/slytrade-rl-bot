"""News integration — Forex Factory economic calendar.

Scrapes Forex Factory calendar for high-impact economic events.
Provides both historical data (for backtesting/RL training) and
live monitoring (for real-time trading).

Events are classified by impact:
- Red (High): NFP, CPI, FOMC, GDP — avoid trading 15min before/after
- Orange (Medium): Retail sales, PMI — caution
- Gray (Low): Minor data — no impact

Usage:
    from slytrade.data.news import ForexFactoryCalendar

    ff = ForexFactoryCalendar()
    events = ff.get_events(date_from="2026-01-01", date_to="2026-08-31")
    high_impact = ff.filter_impact(events, ["High"])
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class NewsEvent:
    """A single economic calendar event."""
    time: datetime           # UTC
    currency: str            # USD, EUR, GBP, etc.
    event: str               # Event name (e.g., "Non-Farm Payrolls")
    impact: str              # High, Medium, Low
    actual: str = ""         # Actual value (after release)
    forecast: str = ""       # Forecast value
    previous: str = ""       # Previous value
    source: str = "forex_factory"

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() in ("high", "red")

    @property
    def is_medium_impact(self) -> bool:
        return self.impact.lower() in ("medium", "orange")

    def to_dict(self) -> dict:
        return {
            "time": self.time.isoformat(),
            "currency": self.currency,
            "event": self.event,
            "impact": self.impact,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
            "source": self.source,
        }


# Known high-impact events with their typical time patterns
# These are used as fallback when scraping fails
KNOWN_HIGH_IMPACT = {
    "USD": [
        "Non-Farm Payrolls",
        "FOMC Rate Decision",
        "CPI m/m",
        "Core CPI m/m",
        "GDP q/q",
        "Retail Sales m/m",
        "ISM Manufacturing PMI",
        "FOMC Press Conference",
        "Fed Chair Powell Speaks",
    ],
    "EUR": [
        "ECB Rate Decision",
        "CPI y/y",
        "GDP q/q",
    ],
    "GBP": [
        "BOE Rate Decision",
        "CPI y/y",
        "GDP q/q",
    ],
}


class ForexFactoryCalendar:
    """Forex Factory economic calendar scraper.

    Scrapes the weekly calendar pages and returns structured event data.
    Caches results to avoid repeated scraping.
    """

    BASE_URL = "https://www.forexfactory.com/calendar"
    CACHE_DIR = Path("data/news")

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _scrape_week(self, week_str: str) -> list[dict]:
        """Scrape one week of calendar data from Forex Factory.

        Uses the JSON API endpoint which is more reliable than HTML parsing.
        Falls back to HTML parsing if JSON fails.

        Args:
            week_str: Week identifier like "jan1.2026" or "aug25.2026"

        Returns:
            List of event dicts
        """
        import ssl
        import urllib.request

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")]

        # Try the calendar page
        url = f"{self.BASE_URL}?week={week_str}"
        try:
            response = opener.open(url, timeout=30)
            html = response.read().decode("utf-8", errors="replace")
        except Exception as e:
            return []

        events = []
        # Parse HTML table rows
        # Forex Factory uses <tr class="calendar__row ..."> for each event
        # Each row has: date, time, currency, event, impact, forecast, previous, actual

        # Split into rows
        row_pattern = re.compile(r'<tr[^>]*class="[^"]*calendar__row[^"]*"[^>]*>(.*?)</tr>', re.DOTALL)
        cell_pattern = re.compile(r'<td[^>]*class="[^"]*calendar__cell[^"]*"[^>]*>(.*?)</td>', re.DOTALL)
        impact_pattern = re.compile(r'calendar__impact[^"]*icon--ff-legend-(\w+)')
        date_pattern = re.compile(r'<span[^>]*class="[^"]*date[^"]*"[^>]*>(.*?)</span>', re.DOTALL)
        time_pattern = re.compile(r'<span[^>]*class="[^"]*time[^"]*"[^>]*>(.*?)</span>', re.DOTALL)
        currency_pattern = re.compile(r'<span[^>]*class="[^"]*currency[^"]*"[^>]*>(.*?)</span>', re.DOTALL)
        event_pattern = re.compile(r'<span[^>]*class="[^"]*event[^"]*"[^>]*>(.*?)</span>', re.DOTALL)

        current_date = ""
        current_year = int(week_str.split(".")[-1]) if "." in week_str else datetime.now().year

        for row_match in row_pattern.finditer(html):
            row_html = row_match.group(1)

            # Extract cells
            cells = cell_pattern.findall(row_html)
            if len(cells) < 5:
                continue

            # Extract date (first cell)
            date_match = date_pattern.search(row_html)
            if date_match:
                raw_date = re.sub(r'<[^>]+>', '', date_match.group(1)).strip()
                if raw_date:
                    current_date = raw_date

            # Extract time
            time_match = time_pattern.search(row_html)
            event_time = ""
            if time_match:
                event_time = re.sub(r'<[^>]+>', '', time_match.group(1)).strip()

            # Extract currency
            currency_match = currency_pattern.search(row_html)
            currency = ""
            if currency_match:
                currency = re.sub(r'<[^>]+>', '', currency_match.group(1)).strip()

            # Extract event name
            event_match = event_pattern.search(row_html)
            event_name = ""
            if event_match:
                event_name = re.sub(r'<[^>]+>', '', event_match.group(1)).strip()

            # Extract impact
            impact_match = impact_pattern.search(row_html)
            impact = "Low"
            if impact_match:
                impact_raw = impact_match.group(1).lower()
                if "high" in impact_raw or "red" in impact_raw:
                    impact = "High"
                elif "medium" in impact_raw or "orange" in impact_raw:
                    impact = "Medium"
                else:
                    impact = "Low"

            # Parse datetime
            if not current_date or not event_name:
                continue

            try:
                # Combine date + time
                dt_str = f"{current_date} {current_year}"
                if event_time and event_time.lower() not in ("all day", "tentative", ""):
                    dt_str = f"{current_date} {current_year} {event_time}"
                    event_dt = datetime.strptime(dt_str, "%b %d %Y %I:%M%p")
                else:
                    event_dt = datetime.strptime(dt_str, "%b %d %Y")
            except ValueError:
                continue

            # Extract actual/forecast/previous from remaining cells
            actual = re.sub(r'<[^>]+>', '', cells[-3]).strip() if len(cells) >= 3 else ""
            forecast = re.sub(r'<[^>]+>', '', cells[-2]).strip() if len(cells) >= 2 else ""
            previous = re.sub(r'<[^>]+>', '', cells[-1]).strip() if len(cells) >= 1 else ""

            events.append({
                "time": event_dt.isoformat(),
                "currency": currency,
                "event": event_name,
                "impact": impact,
                "actual": actual,
                "forecast": forecast,
                "previous": previous,
            })

        return events

    def get_events(
        self,
        date_from: str | datetime,
        date_to: str | datetime,
        currencies: list[str] | None = None,
    ) -> list[NewsEvent]:
        """Get economic events for a date range.

        First checks cache, then scrapes if needed.

        Args:
            date_from: Start date (YYYY-MM-DD or datetime)
            date_to: End date (YYYY-MM-DD or datetime)
            currencies: Filter by currencies (e.g., ["USD", "EUR"])

        Returns:
            List of NewsEvent objects
        """
        if isinstance(date_from, str):
            date_from = datetime.fromisoformat(date_from)
        if isinstance(date_to, str):
            date_to = datetime.fromisoformat(date_to)

        # Check cache first
        cached = self._load_cache(date_from, date_to)
        if cached:
            events = cached
        else:
            # Scrape week by week
            events = []
            current = date_from - timedelta(days=date_from.weekday())  # Start of week
            while current <= date_to:
                week_str = current.strftime("%b%d.%Y").lower()
                week_events = self._scrape_week(week_str)
                events.extend(week_events)
                current += timedelta(days=7)

            # Cache results
            if events:
                self._save_cache(events, date_from, date_to)

        # Convert to NewsEvent objects
        result = []
        for e in events:
            try:
                evt = NewsEvent(
                    time=datetime.fromisoformat(e["time"]) if isinstance(e["time"], str) else e["time"],
                    currency=e.get("currency", ""),
                    event=e.get("event", ""),
                    impact=e.get("impact", "Low"),
                    actual=e.get("actual", ""),
                    forecast=e.get("forecast", ""),
                    previous=e.get("previous", ""),
                )
                result.append(evt)
            except Exception:
                continue

        # Filter by currencies
        if currencies:
            currencies_upper = [c.upper() for c in currencies]
            result = [e for e in result if e.currency.upper() in currencies_upper]

        # Filter by date range
        result = [e for e in result if date_from <= e.time <= date_to]

        return sorted(result, key=lambda e: e.time)

    def filter_impact(self, events: list[NewsEvent], impacts: list[str]) -> list[NewsEvent]:
        """Filter events by impact level."""
        impacts_lower = [i.lower() for i in impacts]
        return [e for e in events if e.impact.lower() in impacts_lower]

    def get_high_impact_windows(
        self,
        events: list[NewsEvent],
        before_minutes: int = 15,
        after_minutes: int = 15,
    ) -> list[tuple[datetime, datetime]]:
        """Get time windows around high-impact events."""
        windows = []
        for e in events:
            if e.is_high_impact:
                start = e.time - timedelta(minutes=before_minutes)
                end = e.time + timedelta(minutes=after_minutes)
                windows.append((start, end))
        return windows

    def is_news_window(
        self,
        timestamp: datetime,
        events: list[NewsEvent],
        before_minutes: int = 15,
        after_minutes: int = 15,
    ) -> tuple[bool, NewsEvent | None]:
        """Check if a timestamp falls within a high-impact news window."""
        for e in events:
            if e.is_high_impact:
                start = e.time - timedelta(minutes=before_minutes)
                end = e.time + timedelta(minutes=after_minutes)
                if start <= timestamp <= end:
                    return True, e
        return False, None

    def _load_cache(self, date_from: datetime, date_to: datetime) -> list[dict] | None:
        """Load cached events for a date range."""
        cache_file = self.cache_dir / f"ff_calendar_{date_from.strftime('%Y%m')}_{date_to.strftime('%Y%m')}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_cache(self, events: list[dict], date_from: datetime, date_to: datetime) -> None:
        """Save events to cache."""
        cache_file = self.cache_dir / f"ff_calendar_{date_from.strftime('%Y%m')}_{date_to.strftime('%Y%m')}.json"
        with open(cache_file, "w") as f:
            json.dump(events, f, indent=2, default=str)


def load_news_parquet(path: str) -> pd.DataFrame:
    """Load news events from a parquet file."""
    return pd.read_parquet(path)


def create_news_features(df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """Add news-related features to a bar DataFrame.

    For each M1 bar, adds:
    - minutes_to_next_high: minutes until next high-impact event
    - minutes_since_last_high: minutes since last high-impact event
    - in_news_window: whether we're within 15min of a high-impact event
    - news_impact_score: numeric impact score (3=High, 2=Medium, 1=Low, 0=None)

    Args:
        df: Bar DataFrame with 'time' column
        news_df: News DataFrame with 'time', 'impact' columns

    Returns:
        DataFrame with news features added
    """
    df = df.copy()
    n = len(df)

    if news_df.empty:
        df["minutes_to_next_high"] = 999.0
        df["minutes_since_last_high"] = 999.0
        df["in_news_window"] = False
        df["news_impact_score"] = 0
        return df

    # Ensure datetime
    news_df = news_df.copy()
    news_df["time"] = pd.to_datetime(news_df["time"], utc=True)
    bar_times = pd.to_datetime(df["time"], utc=True).values

    # Filter high-impact events
    high_impact = news_df[news_df["impact"].str.lower().isin(["high", "red"])].copy()
    high_impact = high_impact.sort_values("time")

    minutes_to_next = np.full(n, 999.0)
    minutes_since_last = np.full(n, 999.0)
    impact_scores = np.zeros(n, dtype=int)

    if not high_impact.empty:
        event_times = high_impact["time"].values
        indices = np.searchsorted(event_times, bar_times)

        for i in range(n):
            idx = indices[i]
            if idx < len(event_times):
                diff = (event_times[idx] - bar_times[i]) / np.timedelta64(1, "m")
                minutes_to_next[i] = max(diff, 0)
            if idx > 0:
                diff = (bar_times[i] - event_times[idx - 1]) / np.timedelta64(1, "m")
                minutes_since_last[i] = max(diff, 0)

    # Impact score for all events (not just high)
    if not news_df.empty:
        impact_map = {"high": 3, "red": 3, "medium": 2, "orange": 2, "low": 1, "gray": 1}
        all_event_times = news_df["time"].values
        all_impacts = news_df["impact"].str.lower().map(impact_map).fillna(0).values
        all_indices = np.searchsorted(all_event_times, bar_times)

        for i in range(n):
            idx = all_indices[i]
            # Check next event within 60 min
            if idx < len(all_event_times):
                diff_min = (all_event_times[idx] - bar_times[i]) / np.timedelta64(1, "m")
                if 0 <= diff_min < 60:
                    impact_scores[i] = max(impact_scores[i], int(all_impacts[idx]))
            # Check previous event within 60 min
            if idx > 0:
                diff_min = (bar_times[i] - all_event_times[idx - 1]) / np.timedelta64(1, "m")
                if 0 <= diff_min < 60:
                    impact_scores[i] = max(impact_scores[i], int(all_impacts[idx - 1]))

    df["minutes_to_next_high"] = minutes_to_next
    df["minutes_since_last_high"] = minutes_since_last
    df["in_news_window"] = (minutes_to_next < 15) | (minutes_since_last < 15)
    df["news_impact_score"] = impact_scores

    return df
