from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from slytrade.data.schemas import BAR_COLUMNS, TICK_COLUMNS
from slytrade.data.time import ensure_utc

OutputFormat = Literal["csv", "parquet"]


def _infer_output_format(path: str | Path) -> OutputFormat:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError("sample output path must end with .csv or .parquet")


def write_sample_frame(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _infer_output_format(path)
    if fmt == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def generate_sample_bars(
    *,
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    start: datetime,
    periods: int = 500,
    seed: int = 42,
    start_price: float = 2400.0,
    trend_per_bar: float = 0.02,
    volatility: float = 0.8,
    spread_points: float = 20.0,
) -> pd.DataFrame:
    """Generate deterministic canonical OHLCV sample bars.

    The sample is not meant to model real market behavior. It exists so tests,
    demos and CLI commands can run without an MT5 terminal.
    """
    if periods <= 0:
        raise ValueError("periods must be positive")
    if start_price <= 0:
        raise ValueError("start_price must be positive")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")

    rng = np.random.default_rng(seed)
    start = ensure_utc(start)
    times = pd.date_range(start, periods=periods, freq="min", tz="UTC")
    shocks = rng.normal(loc=trend_per_bar, scale=volatility, size=periods)
    close = np.maximum(start_price + np.cumsum(shocks), 0.01)
    open_ = np.empty(periods, dtype=float)
    open_[0] = start_price
    open_[1:] = close[:-1]
    wick_up = rng.uniform(0.05, max(volatility, 0.05) + 0.1, size=periods)
    wick_down = rng.uniform(0.05, max(volatility, 0.05) + 0.1, size=periods)
    high = np.maximum(open_, close) + wick_up
    low = np.maximum(np.minimum(open_, close) - wick_down, 0.01)
    tick_volume = rng.integers(50, 500, size=periods)

    frame = pd.DataFrame(
        {
            "time": times,
            "symbol": symbol,
            "timeframe": timeframe,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": tick_volume.astype(float),
            "spread": np.full(periods, spread_points, dtype=float),
            "real_volume": np.zeros(periods, dtype=float),
        }
    )
    return frame[BAR_COLUMNS]


def generate_sample_ticks(
    *,
    symbol: str = "XAUUSD",
    start: datetime,
    periods: int = 2_000,
    seed: int = 42,
    start_price: float = 2400.0,
    tick_interval_ms: int = 1000,
    volatility: float = 0.08,
    spread: float = 0.20,
) -> pd.DataFrame:
    """Generate deterministic canonical tick sample data."""
    if periods <= 0:
        raise ValueError("periods must be positive")
    if start_price <= 0:
        raise ValueError("start_price must be positive")
    if tick_interval_ms <= 0:
        raise ValueError("tick_interval_ms must be positive")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")
    if spread < 0:
        raise ValueError("spread cannot be negative")

    rng = np.random.default_rng(seed)
    start = ensure_utc(start)
    times = pd.date_range(start, periods=periods, freq=f"{tick_interval_ms}ms", tz="UTC")
    mid = np.maximum(start_price + np.cumsum(rng.normal(0.0, volatility, size=periods)), 0.01)
    spread_noise = np.maximum(spread + rng.normal(0.0, spread * 0.05 if spread > 0 else 0.0, size=periods), 0.0)
    bid = mid - spread_noise / 2.0
    ask = mid + spread_noise / 2.0
    volume = rng.integers(1, 20, size=periods)

    frame = pd.DataFrame(
        {
            "time": times.floor("s"),
            "time_msc": times,
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": np.zeros(periods, dtype=float),
            "volume": volume.astype(float),
            "volume_real": np.zeros(periods, dtype=float),
            "flags": np.zeros(periods, dtype=float),
            "spread": ask - bid,
            "mid": mid,
        }
    )
    return frame[TICK_COLUMNS]
