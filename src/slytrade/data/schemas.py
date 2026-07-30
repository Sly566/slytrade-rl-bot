from __future__ import annotations

from typing import Any

import pandas as pd

TICK_COLUMNS = [
    "time",
    "time_msc",
    "symbol",
    "bid",
    "ask",
    "last",
    "volume",
    "volume_real",
    "flags",
    "spread",
    "mid",
]

BAR_COLUMNS = [
    "time",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
]


def _to_datetime_utc(series: pd.Series, *, unit: str | None = None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True)
    return pd.to_datetime(series, unit=unit, utc=True)


def normalize_tick_frame(raw: Any, symbol: str) -> pd.DataFrame:
    """Normalize MT5 tick output into the canonical tick schema."""
    df = pd.DataFrame(raw).copy()
    if df.empty:
        return pd.DataFrame(columns=TICK_COLUMNS)

    if "time" not in df.columns:
        raise ValueError("tick data missing required column: time")

    df["time"] = _to_datetime_utc(df["time"], unit="s")
    if "time_msc" in df.columns:
        df["time_msc"] = _to_datetime_utc(df["time_msc"], unit="ms")
    else:
        df["time_msc"] = df["time"]

    df["symbol"] = symbol
    for column in ["bid", "ask", "last", "volume", "volume_real", "flags"]:
        if column not in df.columns:
            df[column] = 0.0

    for column in ["bid", "ask", "last", "volume", "volume_real", "flags"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["spread"] = df["ask"] - df["bid"]
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    return df[TICK_COLUMNS].sort_values("time_msc").reset_index(drop=True)


def normalize_bar_frame(raw: Any, symbol: str, timeframe: str) -> pd.DataFrame:
    """Normalize MT5 rate/bar output into the canonical bar schema."""
    df = pd.DataFrame(raw).copy()
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)

    required = ["time", "open", "high", "low", "close"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"bar data missing required columns: {missing}")

    df["time"] = _to_datetime_utc(df["time"], unit="s")
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    for column in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df[BAR_COLUMNS].sort_values("time").reset_index(drop=True)
