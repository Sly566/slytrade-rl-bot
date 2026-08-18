"""Offline tick → bar resampling.

The paper loop builds bars incrementally in real time; this module does the
same thing in batch so that a **tick-only** source (e.g. the Exness public
archive) can produce canonical OHLC bars for every timeframe without an MT5
terminal. That is the missing piece for real-data training when the broker
bridge is unavailable.

Bars are built from the tick **mid** price (with bid/ask spread captured as a
statistic), bucketed by the timeframe's calendar grid, and timestamped at bar
open — the same convention as MT5 bars.
"""

from __future__ import annotations

import pandas as pd

from slytrade.data.schemas import BAR_COLUMNS
from slytrade.data.timeframes import TIMEFRAME_DURATIONS


def resample_ticks_to_bars(
    ticks: pd.DataFrame,
    timeframe: str,
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Resample canonical ticks into canonical OHLC bars for ``timeframe``.

    Required tick columns: ``time_msc`` (or ``time``), ``bid``, ``ask``.
    ``mid`` is computed when absent. Returns a frame in canonical ``BAR_COLUMNS``
    order, sorted by bar-open time and timestamped at bar open.
    """
    normalized_tf = timeframe.upper()
    if normalized_tf not in TIMEFRAME_DURATIONS:
        raise ValueError(f"unsupported timeframe: {timeframe}")

    if ticks.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)

    required = {"bid", "ask"}
    missing = required.difference(ticks.columns)
    if missing:
        raise ValueError(f"ticks missing required columns: {sorted(missing)}")

    frame = ticks.copy()
    time_col = "time_msc" if "time_msc" in frame.columns else "time"
    if time_col not in frame.columns:
        raise ValueError("ticks must contain a time column (time_msc or time)")

    frame[time_col] = pd.to_datetime(frame[time_col], utc=True)
    frame = frame.sort_values(time_col).reset_index(drop=True)

    if "mid" not in frame.columns:
        frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0

    resolved_symbol = symbol
    if resolved_symbol is None:
        if "symbol" in frame.columns and frame["symbol"].notna().any():
            resolved_symbol = str(frame["symbol"].dropna().iloc[0])
        else:
            resolved_symbol = "UNKNOWN"

    grouped = frame.set_index(time_col).resample(_pandas_freq(normalized_tf))

    bars = pd.DataFrame(
        {
            "open": grouped["mid"].first(),
            "high": grouped["mid"].max(),
            "low": grouped["mid"].min(),
            "close": grouped["mid"].last(),
            "tick_volume": grouped["mid"].count(),
            "spread": grouped.apply(lambda s: float((s["ask"] - s["bid"]).mean()) if len(s) else 0.0),
        }
    )
    bars = bars.dropna(subset=["open", "high", "low", "close"]).reset_index()
    bars = bars.rename(columns={time_col: "time"})
    bars["time"] = pd.to_datetime(bars["time"], utc=True)
    bars["symbol"] = resolved_symbol
    bars["timeframe"] = normalized_tf
    bars["real_volume"] = 0.0

    for column in ["open", "high", "low", "close", "tick_volume", "spread"]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")

    bars["tick_volume"] = bars["tick_volume"].fillna(0.0).astype(float)
    bars["spread"] = bars["spread"].fillna(0.0)
    return bars[BAR_COLUMNS]


def _pandas_freq(timeframe: str) -> str:
    """Map a SlyTrade timeframe to a pandas resample frequency."""
    return {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1D",
        "W1": "1W",
    }[timeframe]


def resample_bars_to_timeframe(
    bars: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """Resample lower-timeframe bars into a higher timeframe (MT5 conventions).

    Buckets by the timeframe's calendar grid, timestamps at bar open, open =
    first, high/low = max/min, close = last, tick_volume = sum, spread = mean.
    The live paper loop uses this to build the H4/D1 context from the streamed
    decision-timeframe bars so the champion's H4-trend gate runs live exactly as
    it does in backtest (``compute_mtf_ict_features`` needs real higher-TF bars).
    """
    normalized_tf = timeframe.upper()
    if normalized_tf not in TIMEFRAME_DURATIONS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    if bars.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)

    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    frame = bars.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time")
    symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else "UNKNOWN"

    grouped = frame.set_index("time").resample(_pandas_freq(normalized_tf))
    out = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "tick_volume": grouped["tick_volume"].sum() if "tick_volume" in frame.columns else grouped["close"].count(),
            "spread": grouped["spread"].mean() if "spread" in frame.columns else 0.0,
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    out["symbol"] = symbol
    out["timeframe"] = normalized_tf
    out["real_volume"] = 0.0
    for column in ["open", "high", "low", "close", "tick_volume", "spread"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["tick_volume"] = out["tick_volume"].fillna(0.0).astype(float)
    out["spread"] = out["spread"].fillna(0.0)
    return out[BAR_COLUMNS]
