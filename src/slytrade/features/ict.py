from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from slytrade.features.sessions import SESSION_COLUMNS, session_hour_labels

PivotType = Literal["high", "low"]
Direction = Literal["bullish", "bearish"]

BASE_FEATURE_COLUMNS = [
    "atr",
    "atr_norm",
    "volume_ratio",
    "pivot_high_confirmed",
    "pivot_low_confirmed",
    "bos_dir",
    "choch_dir",
    "fvg_bullish",
    "fvg_bearish",
    "fvg_size_atr",
    "nearest_bull_fvg_dist_atr",
    "nearest_bear_fvg_dist_atr",
    "order_block_bullish",
    "order_block_bearish",
    "order_block_strength",
    "nearest_bull_ob_dist_atr",
    "nearest_bear_ob_dist_atr",
    "equal_high",
    "equal_low",
    "liquidity_sweep",
    "premium_discount",
    "price_percentile",
    "trend_strength",
    "distance_from_ema50_atr",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + SESSION_COLUMNS
OUTPUT_COLUMNS = ["time", "symbol", "timeframe", *FEATURE_COLUMNS]


@dataclass(frozen=True)
class ICTFeatureConfig:
    atr_period: int = 14
    pivot_lookback: int = 3
    bos_buffer_atr: float = 0.2
    fvg_min_atr: float = 0.1
    equal_level_tolerance_atr: float = 0.2
    equal_level_max_bars: int = 50
    order_block_search_bars: int = 30
    context_window: int = 100
    ema_fast_period: int = 10
    ema_slow_period: int = 50


@dataclass(frozen=True)
class ConfirmedPivot:
    confirmation_index: int
    pivot_index: int
    price: float
    pivot_type: PivotType


@dataclass(frozen=True)
class FairValueGap:
    index: int
    direction: Direction
    top: float
    bottom: float
    size: float


@dataclass(frozen=True)
class OrderBlock:
    index: int
    direction: Direction
    top: float
    bottom: float
    strength: float
    source_index: int


@dataclass(frozen=True)
class EqualLevel:
    index: int
    level_type: Literal["equal_high", "equal_low"]
    price: float
    first_pivot_index: int
    second_pivot_index: int


def _require_columns(bars: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")


def _numeric_series(bars: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(bars[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=float)
    alpha = 2.0 / (period + 1.0)
    out = np.zeros(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def compute_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute causal Wilder-style ATR from OHLC bars."""
    _require_columns(bars, ["high", "low", "close"])
    high = _numeric_series(bars, "high")
    low = _numeric_series(bars, "low")
    close = _numeric_series(bars, "close")
    n = len(bars)
    if n == 0:
        return pd.Series(dtype=float)

    true_range = np.zeros(n, dtype=float)
    true_range[0] = max(high[0] - low[0], 1e-12)
    for i in range(1, n):
        true_range[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
            1e-12,
        )

    atr = np.zeros(n, dtype=float)
    warmup = min(period, n)
    atr[:warmup] = np.mean(true_range[:warmup])
    for i in range(warmup, n):
        atr[i] = (atr[i - 1] * (period - 1) + true_range[i]) / period
    return pd.Series(atr, index=bars.index, name="atr")


def _confirmed_pivot_at(high: np.ndarray, low: np.ndarray, confirmation_index: int, lookback: int) -> list[ConfirmedPivot]:
    center = confirmation_index - lookback
    if center < lookback or confirmation_index >= len(high):
        return []

    left_start = center - lookback
    right_end = center + lookback
    left_high = high[left_start:center]
    right_high = high[center + 1 : right_end + 1]
    left_low = low[left_start:center]
    right_low = low[center + 1 : right_end + 1]
    if len(left_high) == 0 or len(right_high) == 0:
        return []

    pivots: list[ConfirmedPivot] = []
    if high[center] > float(np.max(left_high)) and high[center] >= float(np.max(right_high)):
        pivots.append(ConfirmedPivot(confirmation_index, center, float(high[center]), "high"))
    if low[center] < float(np.min(left_low)) and low[center] <= float(np.min(right_low)):
        pivots.append(ConfirmedPivot(confirmation_index, center, float(low[center]), "low"))
    return pivots


def _find_order_block(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    volume_sma: np.ndarray,
    break_index: int,
    direction: Direction,
    search_bars: int,
) -> OrderBlock | None:
    start = max(0, break_index - search_bars)
    source_index: int | None = None
    for idx in range(break_index - 1, start - 1, -1):
        bearish = close[idx] < open_[idx]
        bullish = close[idx] > open_[idx]
        if direction == "bullish" and bearish:
            source_index = idx
            break
        if direction == "bearish" and bullish:
            source_index = idx
            break

    if source_index is None:
        return None

    block_top = float(high[source_index])
    block_bottom = float(low[source_index])
    block_range = max(block_top - block_bottom, 1e-12)
    if direction == "bullish":
        displacement = max(float(close[break_index]) - block_top, 0.0)
    else:
        displacement = max(block_bottom - float(close[break_index]), 0.0)
    if displacement <= 0:
        return None

    rel_volume = float(volume[source_index]) / max(float(volume_sma[source_index]), 1e-9)
    displacement_score = min(displacement / (block_range * 5.0), 1.0)
    volume_score = min(rel_volume / 2.0, 1.0)
    strength = (0.65 * displacement_score + 0.35 * volume_score) * 100.0
    return OrderBlock(
        index=break_index,
        direction=direction,
        top=block_top,
        bottom=block_bottom,
        strength=float(strength),
        source_index=source_index,
    )


def compute_ict_features(bars: pd.DataFrame, config: ICTFeatureConfig | None = None) -> pd.DataFrame:
    """Compute causal ICT/SMC features from canonical bars.

    No row uses data from after that row. Confirmed pivots are delayed by
    `pivot_lookback` bars, which prevents lookahead leakage.

    The per-bar state is kept in bounded deques instead of rescanning the whole
    pivot/FVG/order-block history, so the whole pass is O(n) in the bar count
    (the old implementation rescanned every past pivot per bar — O(n^2) —
    which made multi-year alignment take hours).
    """
    cfg = config or ICTFeatureConfig()
    _require_columns(bars, ["time", "symbol", "timeframe", "open", "high", "low", "close"])
    data = bars.sort_values("time").reset_index(drop=True).copy()
    n = len(data)
    if n == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    open_ = _numeric_series(data, "open")
    high = _numeric_series(data, "high")
    low = _numeric_series(data, "low")
    close = _numeric_series(data, "close")
    volume = _numeric_series(data, "tick_volume") if "tick_volume" in data.columns else np.ones(n, dtype=float)
    volume_sma = pd.Series(volume).rolling(cfg.atr_period, min_periods=1).mean().to_numpy(dtype=float)
    atr = compute_atr(data, cfg.atr_period).to_numpy(dtype=float)
    ema_fast = _ema(close, cfg.ema_fast_period)
    ema_slow = _ema(close, cfg.ema_slow_period)

    # Vectorized context-window extrema (same [i-context_window+1, i] window).
    context_high = pd.Series(high).rolling(cfg.context_window, min_periods=1).max().to_numpy(dtype=float)
    context_low = pd.Series(low).rolling(cfg.context_window, min_periods=1).min().to_numpy(dtype=float)

    # Vectorized session one-hots (UTC hour-of-day, same ranges as session_label).
    session_cols = session_hour_labels(pd.to_datetime(data["time"], utc=True).dt.hour.to_numpy())

    # Bounded state. maxlen=10 mirrors the old "last 10 pivots of each type"
    # equal-level scan exactly; the window deques mirror the old per-bar
    # filtering of the unbounded pivot/FVG/OB/equal-level lists.
    high_pivots: deque[tuple[int, float]] = deque(maxlen=10)  # (pivot_index, price)
    low_pivots: deque[tuple[int, float]] = deque(maxlen=10)
    win_highs: deque[tuple[int, float]] = deque()  # (confirmation_index, price)
    win_lows: deque[tuple[int, float]] = deque()
    active_fvgs: deque[FairValueGap] = deque()
    active_obs: deque[OrderBlock] = deque()
    equal_levels: deque[EqualLevel] = deque()
    last_high: ConfirmedPivot | None = None
    last_low: ConfirmedPivot | None = None
    trend = 0

    out = {column: np.zeros(n, dtype=float) for column in FEATURE_COLUMNS}

    for i in range(n):
        atr_i = max(float(atr[i]), 1e-12)
        close_i = max(float(close[i]), 1e-12)
        start = max(0, i - cfg.context_window + 1)

        # --- Confirmed pivots + equal-level detection (bounded) ------------
        pivot_high_confirmed = 0.0
        pivot_low_confirmed = 0.0
        for pivot in _confirmed_pivot_at(high, low, i, cfg.pivot_lookback):
            if pivot.pivot_type == "high":
                pivot_high_confirmed = 1.0
                for prev_index, prev_price in reversed(tuple(high_pivots)):
                    if pivot.pivot_index - prev_index <= cfg.equal_level_max_bars:
                        if abs(pivot.price - prev_price) <= cfg.equal_level_tolerance_atr * atr_i:
                            equal_levels.append(
                                EqualLevel(
                                    i,
                                    "equal_high",
                                    (pivot.price + prev_price) / 2.0,
                                    prev_index,
                                    pivot.pivot_index,
                                )
                            )
                            break
                last_high = pivot
                high_pivots.append((pivot.pivot_index, pivot.price))
                win_highs.append((i, pivot.price))
            else:
                pivot_low_confirmed = 1.0
                for prev_index, prev_price in reversed(tuple(low_pivots)):
                    if pivot.pivot_index - prev_index <= cfg.equal_level_max_bars:
                        if abs(pivot.price - prev_price) <= cfg.equal_level_tolerance_atr * atr_i:
                            equal_levels.append(
                                EqualLevel(
                                    i,
                                    "equal_low",
                                    (pivot.price + prev_price) / 2.0,
                                    prev_index,
                                    pivot.pivot_index,
                                )
                            )
                            break
                last_low = pivot
                low_pivots.append((pivot.pivot_index, pivot.price))
                win_lows.append((i, pivot.price))

        # Prune the windowed state to the causal context window (nothing added
        # this bar can be pruned: its index/confirmation_index == i >= start).
        while win_highs and win_highs[0][0] < start:
            win_highs.popleft()
        while win_lows and win_lows[0][0] < start:
            win_lows.popleft()
        while active_fvgs and active_fvgs[0].index < start:
            active_fvgs.popleft()
        while active_obs and active_obs[0].index < start:
            active_obs.popleft()
        while equal_levels and equal_levels[0].index < start:
            equal_levels.popleft()

        # --- Fair value gaps -------------------------------------------------
        fvg_bullish = 0.0
        fvg_bearish = 0.0
        fvg_size_atr = 0.0
        if i >= 2:
            if low[i] > high[i - 2]:
                size = float(low[i] - high[i - 2])
                if size >= cfg.fvg_min_atr * atr_i:
                    fvg_bullish = 1.0
                    fvg_size_atr = size / atr_i
                    active_fvgs.append(FairValueGap(i, "bullish", float(low[i]), float(high[i - 2]), size))
            elif high[i] < low[i - 2]:
                size = float(low[i - 2] - high[i])
                if size >= cfg.fvg_min_atr * atr_i:
                    fvg_bearish = 1.0
                    fvg_size_atr = size / atr_i
                    active_fvgs.append(FairValueGap(i, "bearish", float(low[i - 2]), float(high[i]), size))

        # --- Structure breaks / trend / order blocks ------------------------
        bos_dir = 0.0
        choch_dir = 0.0
        order_block_bullish = 0.0
        order_block_bearish = 0.0
        order_block_strength = 0.0
        buffer = cfg.bos_buffer_atr * atr_i
        if last_high is not None and close[i] > last_high.price + buffer:
            bos_dir = 1.0
            if trend == -1:
                choch_dir = 1.0
            trend = 1
            block = _find_order_block(open_, high, low, close, volume, volume_sma, i, "bullish", cfg.order_block_search_bars)
            if block is not None:
                active_obs.append(block)
                order_block_bullish = 1.0
                order_block_strength = block.strength / 100.0
        elif last_low is not None and close[i] < last_low.price - buffer:
            bos_dir = -1.0
            if trend == 1:
                choch_dir = -1.0
            trend = -1
            block = _find_order_block(open_, high, low, close, volume, volume_sma, i, "bearish", cfg.order_block_search_bars)
            if block is not None:
                active_obs.append(block)
                order_block_bearish = 1.0
                order_block_strength = block.strength / 100.0

        # --- Nearest active FVG distances ------------------------------------
        nearest_bull_fvg_dist = 0.0
        nearest_bear_fvg_dist = 0.0
        for gap in active_fvgs:
            if gap.direction == "bullish":
                distance = close_i - gap.bottom
                if distance >= 0:
                    normalized = distance / atr_i
                    nearest_bull_fvg_dist = normalized if nearest_bull_fvg_dist == 0.0 else min(nearest_bull_fvg_dist, normalized)
            else:
                distance = gap.top - close_i
                if distance >= 0:
                    normalized = distance / atr_i
                    nearest_bear_fvg_dist = normalized if nearest_bear_fvg_dist == 0.0 else min(nearest_bear_fvg_dist, normalized)

        # --- Nearest active order-block distances ---------------------------
        nearest_bull_ob_dist = 0.0
        nearest_bear_ob_dist = 0.0
        max_active_ob_strength = order_block_strength
        for block in active_obs:
            max_active_ob_strength = max(max_active_ob_strength, block.strength / 100.0)
            if block.direction == "bullish":
                distance = close_i - block.bottom
                if distance >= 0:
                    normalized = distance / atr_i
                    nearest_bull_ob_dist = normalized if nearest_bull_ob_dist == 0.0 else min(nearest_bull_ob_dist, normalized)
            else:
                distance = block.top - close_i
                if distance >= 0:
                    normalized = distance / atr_i
                    nearest_bear_ob_dist = normalized if nearest_bear_ob_dist == 0.0 else min(nearest_bear_ob_dist, normalized)

        # --- Equal levels + liquidity sweep ----------------------------------
        equal_high = 1.0 if equal_levels and equal_levels[-1].level_type == "equal_high" and equal_levels[-1].index == i else 0.0
        equal_low = 1.0 if equal_levels and equal_levels[-1].level_type == "equal_low" and equal_levels[-1].index == i else 0.0
        liquidity_sweep = 0.0
        recent = tuple(reversed(tuple(equal_levels)[-20:]))
        for level in recent:
            if level.level_type == "equal_high" and high[i] > level.price and close[i] < level.price:
                liquidity_sweep = 1.0
                break
            if level.level_type == "equal_low" and low[i] < level.price and close[i] > level.price:
                liquidity_sweep = -1.0
                break

        # --- Premium / discount from confirmed pivots in the window ----------
        premium_discount = 0.0
        if win_highs and win_lows:
            max_high = max(price for _, price in win_highs)
            min_low = min(price for _, price in win_lows)
            if max_high > min_low:
                premium_discount = ((close_i - min_low) / (max_high - min_low) - 0.5) * 2.0

        price_percentile = (close_i - context_low[i]) / max(context_high[i] - context_low[i], 1e-12)
        trend_strength = (float(ema_fast[i]) - float(ema_slow[i])) / atr_i
        distance_from_ema50 = (close_i - float(ema_slow[i])) / atr_i
        volume_ratio = float(volume[i]) / max(float(volume_sma[i]), 1e-9)

        out["atr"][i] = atr_i
        out["atr_norm"][i] = atr_i / close_i
        out["volume_ratio"][i] = volume_ratio
        out["pivot_high_confirmed"][i] = pivot_high_confirmed
        out["pivot_low_confirmed"][i] = pivot_low_confirmed
        out["bos_dir"][i] = bos_dir
        out["choch_dir"][i] = choch_dir
        out["fvg_bullish"][i] = fvg_bullish
        out["fvg_bearish"][i] = fvg_bearish
        out["fvg_size_atr"][i] = fvg_size_atr
        out["nearest_bull_fvg_dist_atr"][i] = nearest_bull_fvg_dist
        out["nearest_bear_fvg_dist_atr"][i] = nearest_bear_fvg_dist
        out["order_block_bullish"][i] = order_block_bullish
        out["order_block_bearish"][i] = order_block_bearish
        out["order_block_strength"][i] = max_active_ob_strength
        out["nearest_bull_ob_dist_atr"][i] = nearest_bull_ob_dist
        out["nearest_bear_ob_dist_atr"][i] = nearest_bear_ob_dist
        out["equal_high"][i] = equal_high
        out["equal_low"][i] = equal_low
        out["liquidity_sweep"][i] = liquidity_sweep
        out["premium_discount"][i] = premium_discount
        out["price_percentile"][i] = price_percentile
        out["trend_strength"][i] = trend_strength
        out["distance_from_ema50_atr"][i] = distance_from_ema50

    for column in SESSION_COLUMNS:
        out[column] = session_cols[column]

    result = pd.DataFrame(
        {
            "time": data["time"].to_numpy(),
            "symbol": data["symbol"].astype(str).to_numpy(),
            "timeframe": data["timeframe"].astype(str).to_numpy(),
            **out,
        }
    )
    return result[OUTPUT_COLUMNS]
