"""Tick-derived features for M1 bars.

Computes microstructure features from raw tick data and merges them
into M1 bars. These features give the RL agent visibility into
intra-bar price dynamics that bars alone cannot show.

Features computed per M1 bar:
- tick_buy_ratio: fraction of ticks that are buy-initiated (from flags)
- tick_sell_ratio: fraction of ticks that are sell-initiated
- tick_spread_mean: average bid-ask spread during the bar
- tick_spread_max: maximum spread during the bar (liquidity events)
- tick_spread_std: spread volatility (uncertainty indicator)
- tick_price_velocity: price range / tick count (momentum speed)
- tick_volume_imbalance: (buy_vol - sell_vol) / total_vol (directional pressure)
- tick_absorption: volume / price_range (high vol + small move = absorption)
- tick_count: number of ticks in the bar (activity level)
- tick_buy_volume: total buy-initiated volume
- tick_sell_volume: total sell-initiated volume
- tick_large_trade_ratio: fraction of volume from large trades (>2x median)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# MT5 tick flags
TICK_FLAG_BUY = 0x02
TICK_FLAG_SELL = 0x04


def load_tick_data(tick_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load tick data files covering the date range.

    Ticks are stored as: tick_dir/year=YYYY/month=MM/day=DD.parquet
    """
    tick_files = sorted(tick_dir.rglob("*.parquet"))
    if not tick_files:
        return pd.DataFrame()

    # Filter files by date range
    relevant_files = []
    for f in tick_files:
        # Parse date from path: year=YYYY/month=MM/day=DD.parquet
        try:
            parts = f.stem  # "DD"
            parent = f.parent  # month=MM
            gparent = parent.parent  # year=YYYY
            year = int(gparent.name.replace("year=", ""))
            month = int(parent.name.replace("month=", ""))
            day = int(parts.replace("day=", ""))
            file_date = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
            # Ensure start/end are tz-aware for comparison
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
            # Include files within range + 1 day buffer on each side
            if (start - pd.Timedelta(days=1)) <= file_date <= (end + pd.Timedelta(days=1)):
                relevant_files.append(f)
        except (ValueError, IndexError):
            continue

    if not relevant_files:
        return pd.DataFrame()

    frames = []
    for f in relevant_files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    ticks = pd.concat(frames, ignore_index=True)
    del frames

    # Ensure time column
    if "time_msc" in ticks.columns:
        ticks["time"] = pd.to_datetime(ticks["time_msc"], unit="ms", utc=True)
    elif "time" in ticks.columns:
        ticks["time"] = pd.to_datetime(ticks["time"], utc=True)

    ticks = ticks.sort_values("time").reset_index(drop=True)
    return ticks


def compute_tick_features(
    m1_bars: pd.DataFrame,
    tick_dir: Path,
) -> pd.DataFrame:
    """Compute tick-derived features and merge into M1 bars.

    For each M1 bar, aggregates all ticks that fall within the bar's
    time window [bar_open, bar_open + 1min).

    Args:
        m1_bars: M1 OHLCV bars with 'time' column
        tick_dir: Path to tick data directory (e.g., data/raw/mt5_ticks/symbol=XAUUSDm/)

    Returns:
        DataFrame with tick feature columns added (same index as m1_bars)
    """
    if not tick_dir.exists():
        # No tick data — return zeros
        return _empty_tick_features(m1_bars)

    # Load ticks for the date range of M1 bars
    start = m1_bars["time"].min()
    end = m1_bars["time"].max()

    try:
        ticks = load_tick_data(tick_dir, start, end)
    except Exception:
        return _empty_tick_features(m1_bars)
    if ticks.empty:
        return _empty_tick_features(m1_bars)

    # Ensure numeric columns
    for col in ["bid", "ask", "volume", "volume_real", "flags", "spread", "mid"]:
        if col in ticks.columns:
            ticks[col] = pd.to_numeric(ticks[col], errors="coerce").fillna(0.0)
        else:
            ticks[col] = 0.0

    # Classify ticks as buy/sell from flags
    flags = ticks["flags"].values.astype(np.int64)
    ticks["is_buy"] = (flags & TICK_FLAG_BUY) > 0
    ticks["is_sell"] = (flags & TICK_FLAG_SELL) > 0
    # If neither flag set, classify by price movement
    no_flag = ~ticks["is_buy"] & ~ticks["is_sell"]
    if no_flag.any() and "mid" in ticks.columns:
        mid_diff = ticks["mid"].diff().fillna(0)
        ticks.loc[no_flag & (mid_diff > 0), "is_buy"] = True
        ticks.loc[no_flag & (mid_diff < 0), "is_sell"] = True

    # Assign volume to buy/sell
    vol = ticks["volume_real"].where(ticks["volume_real"] > 0, ticks["volume"])
    ticks["buy_vol"] = vol * ticks["is_buy"].astype(float)
    ticks["sell_vol"] = vol * ticks["is_sell"].astype(float)

    # Create M1 bar time boundaries
    bar_times = m1_bars["time"].values
    n = len(m1_bars)

    # Pre-allocate output arrays
    tick_buy_ratio = np.zeros(n, dtype=np.float64)
    tick_sell_ratio = np.zeros(n, dtype=np.float64)
    tick_spread_mean = np.zeros(n, dtype=np.float64)
    tick_spread_max = np.zeros(n, dtype=np.float64)
    tick_spread_std = np.zeros(n, dtype=np.float64)
    tick_price_velocity = np.zeros(n, dtype=np.float64)
    tick_volume_imbalance = np.zeros(n, dtype=np.float64)
    tick_absorption = np.zeros(n, dtype=np.float64)
    tick_count_arr = np.zeros(n, dtype=np.float64)
    tick_buy_volume = np.zeros(n, dtype=np.float64)
    tick_sell_volume = np.zeros(n, dtype=np.float64)
    tick_large_trade_ratio = np.zeros(n, dtype=np.float64)

    tick_times = ticks["time"].values
    tick_bids = ticks["bid"].values
    tick_spreads = ticks["spread"].values
    tick_buy_vols = ticks["buy_vol"].values
    tick_sell_vols = ticks["sell_vol"].values
    tick_vols = (ticks["volume_real"].where(ticks["volume_real"] > 0, ticks["volume"])).values

    # Use searchsorted for efficient tick-to-bar assignment
    bar_starts = bar_times
    # Bar end = next bar start (or + 1 minute for last bar)
    if n > 1:
        bar_ends = np.append(bar_times[1:], bar_times[-1] + np.timedelta64(1, "m"))
    else:
        bar_ends = bar_times + np.timedelta64(1, "m")

    # Median volume for large trade detection
    median_vol = np.median(tick_vols[tick_vols > 0]) if np.any(tick_vols > 0) else 1.0

    for i in range(n):
        # Find ticks in this bar's time window
        lo = np.searchsorted(tick_times, bar_starts[i], side="left")
        hi = np.searchsorted(tick_times, bar_ends[i], side="left")

        if lo >= hi:
            continue

        bar_bids = tick_bids[lo:hi]
        bar_spreads = tick_spreads[lo:hi]
        bar_bv = tick_buy_vols[lo:hi]
        bar_sv = tick_sell_vols[lo:hi]
        bar_vol = tick_vols[lo:hi]
        n_ticks = hi - lo

        tick_count_arr[i] = n_ticks

        # Buy/sell ratios
        total_bv = bar_bv.sum()
        total_sv = bar_sv.sum()
        total_vol = total_bv + total_sv
        tick_buy_volume[i] = total_bv
        tick_sell_volume[i] = total_sv

        if total_vol > 0:
            tick_buy_ratio[i] = total_bv / total_vol
            tick_sell_ratio[i] = total_sv / total_vol
            tick_volume_imbalance[i] = (total_bv - total_sv) / total_vol

        # Spread dynamics
        if len(bar_spreads) > 0:
            tick_spread_mean[i] = np.mean(bar_spreads)
            tick_spread_max[i] = np.max(bar_spreads)
            tick_spread_std[i] = np.std(bar_spreads) if len(bar_spreads) > 1 else 0.0

        # Price velocity: price range / tick count
        if len(bar_bids) > 1 and n_ticks > 1:
            price_range = np.max(bar_bids) - np.min(bar_bids)
            tick_price_velocity[i] = price_range / n_ticks

            # Absorption: volume / price_range (high vol + small move)
            if price_range > 0:
                tick_absorption[i] = total_vol / price_range

        # Large trade ratio
        if len(bar_vol) > 0 and median_vol > 0:
            large_mask = bar_vol > 2 * median_vol
            if bar_vol.sum() > 0:
                tick_large_trade_ratio[i] = bar_vol[large_mask].sum() / bar_vol.sum()

    # Normalize by ATR where appropriate
    atr = m1_bars.get("atr_14", pd.Series(1.0, index=m1_bars.index)).values
    atr = np.where(np.isnan(atr) | (atr <= 0), 1.0, atr)

    # Assemble output
    result = pd.DataFrame({
        "tick_buy_ratio": tick_buy_ratio,
        "tick_sell_ratio": tick_sell_ratio,
        "tick_spread_mean": tick_spread_mean,
        "tick_spread_max": tick_spread_max,
        "tick_spread_std": tick_spread_std,
        "tick_price_velocity": tick_price_velocity / atr,  # normalize by ATR
        "tick_volume_imbalance": tick_volume_imbalance,
        "tick_absorption": np.log1p(tick_absorption) / 10.0,  # log-scale, divide by 10
        "tick_count": tick_count_arr,
        "tick_buy_volume": tick_buy_volume,
        "tick_sell_volume": tick_sell_volume,
        "tick_large_trade_ratio": tick_large_trade_ratio,
    }, index=m1_bars.index)

    return result


def _empty_tick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of zeros when no tick data available."""
    n = len(df)
    return pd.DataFrame({
        "tick_buy_ratio": np.zeros(n),
        "tick_sell_ratio": np.zeros(n),
        "tick_spread_mean": np.zeros(n),
        "tick_spread_max": np.zeros(n),
        "tick_spread_std": np.zeros(n),
        "tick_price_velocity": np.zeros(n),
        "tick_volume_imbalance": np.zeros(n),
        "tick_absorption": np.zeros(n),
        "tick_count": np.zeros(n),
        "tick_buy_volume": np.zeros(n),
        "tick_sell_volume": np.zeros(n),
        "tick_large_trade_ratio": np.zeros(n),
    }, index=df.index)
