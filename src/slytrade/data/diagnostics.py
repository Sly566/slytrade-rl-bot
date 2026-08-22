"""Sanity checks on raw bars and ticks — cheap diagnostics for Layer 1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .time import add_decision_time


@dataclass
class BarDiagnostics:
    rows: int = 0
    start: str = ""
    end: str = ""
    duplicates: int = 0
    invalid_ohlc: int = 0
    issues: list[str] = field(default_factory=list)


@dataclass
class TickDiagnostics:
    rows: int = 0
    start: str = ""
    end: str = ""
    duplicates: int = 0
    bad_prices: int = 0
    crossed_spread: int = 0
    issues: list[str] = field(default_factory=list)


def inspect_bars(bars: pd.DataFrame, timeframe: str | None = None) -> BarDiagnostics:
    diag = BarDiagnostics()
    if bars is None or bars.empty:
        diag.issues.append("empty frame")
        return diag

    required = {"time", "open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        diag.issues.append(f"missing columns: {sorted(missing)}")
        return diag

    df = add_decision_time(bars, tf=timeframe) if "decision_time" not in bars.columns else bars.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    diag.rows = len(df)
    diag.start = str(df["time"].min())
    diag.end = str(df["time"].max())
    diag.duplicates = int(df.duplicated(subset=["time"]).sum())

    invalid_mask = (
        (df["high"] < df["low"])
        | (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
    )
    diag.invalid_ohlc = int(invalid_mask.sum())

    if diag.duplicates:
        diag.issues.append(f"{diag.duplicates} duplicate timestamps")
    if diag.invalid_ohlc:
        diag.issues.append(f"{diag.invalid_ohlc} rows with invalid OHLC")
    if not df["time"].is_monotonic_increasing:
        diag.issues.append("timestamps not monotonic increasing")
    return diag


def inspect_ticks(ticks: pd.DataFrame) -> TickDiagnostics:
    diag = TickDiagnostics()
    if ticks is None or ticks.empty:
        diag.issues.append("empty frame")
        return diag

    required = {"time_msc", "bid", "ask"}
    missing = required - set(ticks.columns)
    if missing:
        diag.issues.append(f"missing columns: {sorted(missing)}")
        return diag

    df = ticks.copy()
    df["time_msc"] = pd.to_datetime(df["time_msc"], utc=True, errors="coerce")
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df = df.dropna(subset=["time_msc"])

    diag.rows = len(df)
    diag.start = str(df["time_msc"].min())
    diag.end = str(df["time_msc"].max())
    diag.duplicates = int(df.duplicated(subset=["time_msc"]).sum())
    diag.bad_prices = int(((df["bid"] <= 0) | (df["ask"] <= 0)).sum())
    diag.crossed_spread = int((df["ask"] < df["bid"]).sum())

    if diag.duplicates:
        diag.issues.append(f"{diag.duplicates} duplicate time_msc")
    if diag.bad_prices:
        diag.issues.append(f"{diag.bad_prices} rows with bid/ask <= 0")
    if diag.crossed_spread:
        diag.issues.append(f"{diag.crossed_spread} rows with ask < bid")
    if not df["time_msc"].is_monotonic_increasing:
        diag.issues.append("time_msc not monotonic increasing")
    return diag
