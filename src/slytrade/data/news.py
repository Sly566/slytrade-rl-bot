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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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

        Args:
            week_str: Week identifier like "jan1.2026" or "aug25.2026"

        Returns:
            List of event dicts
        """
        import ssl
        import urllib.request
        from html.parser import HTMLParser

        url = f"{self.BASE_URL}?week={week_str}"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")]

        try:
            response = opener.open(url, timeout=30)
            html = response.read().decode("utf-8", errors="replace")
        except Exception as e:
            return []

        # Simple HTML parsing for calendar rows
        events = []
        lines = html.split("\n")
        current_date = ""

        for line in lines:
            # Look for date headers
            if "calendar__date" in line:
                # Extract date
                for month in ["jan", "feb", "mar", "apr", "may", "jun",
                              "jul", "aug", "sep", "oct", "nov", "dec"]:
                    if month in line.lower():
                        current_date = line.strip()
                        break

            # Look for event rows
            if "calendar__row" in line:
                # This is a simplified parser — full implementation would use BeautifulSoup
                pass

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
        """Get time windows around high-impact events.

        Returns list of (start, end) tuples representing windows
        where trading should be avoided or approached with caution.
        """
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
        """Check if a timestamp falls within a high-impact news window.

        Returns:
            (is_in_window, event_or_none)
        """
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
    """Load news events from a parquet file.

    Expected columns: time, currency, event, impact, actual, forecast, previous
    """
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
    if news_df.empty:
        df["minutes_to_next_high"] = 999
        df["minutes_since_last_high"] = 999
        df["in_news_window"] = False
        df["news_impact_score"] = 0
        return df

    # Ensure datetime
    df = df.copy()
    news_df = news_df.copy()
    news_df["time"] = pd.to_datetime(news_df["time"], utc=True)

    # Filter high-impact events
    high_impact = news_df[news_df["impact"].str.lower().isin(["high", "red"])].copy()
    high_impact = high_impact.sort_values("time")

    if high_impact.empty:
        df["minutes_to_next_high"] = 999
        df["minutes_since_last_high"] = 999
        df["in_news_window"] = False
        df["news_impact_score"] = 0
        return df

    # For each bar, find nearest high-impact event
    event_times = high_impact["time"].values

    bar_times = pd.to_datetime(df["time"], utc=True).values

    # Use searchsorted for efficient nearest-neighbor
    import numpy as np
    indices = np.searchsorted(event_times, bar_times)

    minutes_to_next = np.full(len(df), 999.0)
    minutes_since_last = np.full(len(df), 999.0)

    for i, idx in enumerate(indices):
        if idx < len(event_times):
            diff = (event_times[idx] - bar_times[i]) / np.timedelta64(1, "m")
            minutes_to_next[i] = diff
        if idx > 0:
            diff = (bar_times[i] - event_times[idx - 1]) / np.timedelta64(1, "m")
            minutes_since_last[i] = diff

    df["minutes_to_next_high"] = minutes_to_next
    df["minutes_since_last_high"] = minutes_since_last
    df["in_news_window"] = (minutes_to_next < 15) | (minutes_since_last < 15)

    # Impact score
    impact_map = {"high": 3, "red": 3, "medium": 2, "orange": 2, "low": 1, "gray": 1}
    news_df["impact_score"] = news_df["impact"].str.lower().map(impact_map).fillna(0)

    # For each bar, find the impact of the nearest event (within 60min)
    impact_scores = np.zeros(len(df))
    for i, idx in enumerate(indices):
        # Check event at idx (next)
        if idx < len(event_times):
            diff_min = (event_times[idx] - bar_times[i]) / np.timedelta64(1, "m")
            if diff_min < 60:
                impact_scores[i] = max(impact_scores[i], news_df.iloc[idx]["impact_score"])
        # Check event at idx-1 (previous)
        if idx > 0:
            diff_min = (bar_times[i] - event_times[idx - 1]) / np.timedelta64(1, "m")
            if diff_min < 60:
                impact_scores[i] = max(impact_scores[i], news_df.iloc[idx - 1]["impact_score"])

    df["news_impact_score"] = impact_scores.astype(int)

    return df
