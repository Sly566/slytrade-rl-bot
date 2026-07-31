from __future__ import annotations

from datetime import timedelta

import pandas as pd

TIMEFRAME_DURATIONS: dict[str, timedelta] = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
    "D1": timedelta(days=1),
    "W1": timedelta(days=7),
}


def timeframe_duration(timeframe: str) -> timedelta:
    normalized = timeframe.upper()
    try:
        return TIMEFRAME_DURATIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from exc


def add_decision_time(bars: pd.DataFrame, *, timeframe: str | None = None) -> pd.DataFrame:
    """Add a causal decision_time column to canonical bars.

    MT5 bars are timestamped at bar open. If a strategy uses the full OHLC bar,
    the earliest causal decision time is bar open + timeframe duration.
    """
    if "time" not in bars.columns:
        raise ValueError("bars missing required column: time")
    result = bars.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True)

    if timeframe is None:
        if "timeframe" not in result.columns or result.empty:
            raise ValueError("timeframe must be provided when bars lack a timeframe column")
        timeframes = sorted(str(value).upper() for value in result["timeframe"].dropna().unique())
        if len(timeframes) != 1:
            raise ValueError(f"expected one timeframe, found {timeframes}; pass timeframe explicitly")
        timeframe = timeframes[0]

    duration = timeframe_duration(timeframe)
    result["decision_time"] = result["time"] + duration
    return result


def decision_time_for_bar(bar: pd.Series, *, default_timeframe: str | None = None) -> pd.Timestamp:
    """Return the causal decision timestamp for a bar row."""
    if "decision_time" in bar.index and pd.notna(bar["decision_time"]):
        return pd.Timestamp(bar["decision_time"]).tz_convert("UTC") if pd.Timestamp(bar["decision_time"]).tzinfo else pd.Timestamp(bar["decision_time"], tz="UTC")
    timeframe = default_timeframe or str(bar.get("timeframe", ""))
    if not timeframe:
        raise ValueError("bar has no decision_time and no timeframe")
    bar_time = pd.Timestamp(bar["time"])
    if bar_time.tzinfo is None:
        bar_time = bar_time.tz_localize("UTC")
    else:
        bar_time = bar_time.tz_convert("UTC")
    return bar_time + timeframe_duration(timeframe)
