from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from slytrade.features.sessions import SESSION_COLUMNS, session_one_hot

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

    pivots: list[ConfirmedPivot] = []
    active_fvgs: list[FairValueGap] = []
    active_obs: list[OrderBlock] = []
    equal_levels: list[EqualLevel] = []
    last_high: ConfirmedPivot | None = None
    last_low: ConfirmedPivot | None = None
    trend = 0

    feature_rows: list[dict[str, float | str | pd.Timestamp]] = []

    for i in range(n):
        atr_i = max(float(atr[i]), 1e-12)
        close_i = max(float(close[i]), 1e-12)
        start = max(0, i - cfg.context_window + 1)

        pivot_high_confirmed = 0.0
        pivot_low_confirmed = 0.0
        for pivot in _confirmed_pivot_at(high, low, i, cfg.pivot_lookback):
            pivots.append(pivot)
            if pivot.pivot_type == "high":
                pivot_high_confirmed = 1.0
                prior = [p for p in pivots[:-1] if p.pivot_type == "high"]
                for prev in reversed(prior[-10:]):
                    if pivot.pivot_index - prev.pivot_index <= cfg.equal_level_max_bars:
                        if abs(pivot.price - prev.price) <= cfg.equal_level_tolerance_atr * atr_i:
                            equal_levels.append(EqualLevel(i, "equal_high", (pivot.price + prev.price) / 2.0, prev.pivot_index, pivot.pivot_index))
                            break
                last_high = pivot
            else:
                pivot_low_confirmed = 1.0
                prior = [p for p in pivots[:-1] if p.pivot_type == "low"]
                for prev in reversed(prior[-10:]):
                    if pivot.pivot_index - prev.pivot_index <= cfg.equal_level_max_bars:
                        if abs(pivot.price - prev.price) <= cfg.equal_level_tolerance_atr * atr_i:
                            equal_levels.append(EqualLevel(i, "equal_low", (pivot.price + prev.price) / 2.0, prev.pivot_index, pivot.pivot_index))
                            break
                last_low = pivot

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

        active_fvgs = [gap for gap in active_fvgs if gap.index >= start]
        active_obs = [block for block in active_obs if block.index >= start]

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

        recent_equal = [level for level in equal_levels if level.index >= start]
        equal_high = 1.0 if recent_equal and recent_equal[-1].level_type == "equal_high" and recent_equal[-1].index == i else 0.0
        equal_low = 1.0 if recent_equal and recent_equal[-1].level_type == "equal_low" and recent_equal[-1].index == i else 0.0
        liquidity_sweep = 0.0
        for level in reversed(recent_equal[-20:]):
            if level.level_type == "equal_high" and high[i] > level.price and close[i] < level.price:
                liquidity_sweep = 1.0
                break
            if level.level_type == "equal_low" and low[i] < level.price and close[i] > level.price:
                liquidity_sweep = -1.0
                break

        confirmed_pivots = [pivot for pivot in pivots if start <= pivot.confirmation_index <= i]
        pivot_highs = [pivot.price for pivot in confirmed_pivots if pivot.pivot_type == "high"]
        pivot_lows = [pivot.price for pivot in confirmed_pivots if pivot.pivot_type == "low"]
        premium_discount = 0.0
        if pivot_highs and pivot_lows and max(pivot_highs) > min(pivot_lows):
            premium_discount = ((close_i - min(pivot_lows)) / (max(pivot_highs) - min(pivot_lows)) - 0.5) * 2.0

        context_high = float(np.max(high[start : i + 1]))
        context_low = float(np.min(low[start : i + 1]))
        price_percentile = (close_i - context_low) / max(context_high - context_low, 1e-12)
        trend_strength = (float(ema_fast[i]) - float(ema_slow[i])) / atr_i
        distance_from_ema50 = (close_i - float(ema_slow[i])) / atr_i
        volume_ratio = float(volume[i]) / max(float(volume_sma[i]), 1e-9)

        row: dict[str, float | str | pd.Timestamp] = {
            "time": data.loc[i, "time"],
            "symbol": str(data.loc[i, "symbol"]),
            "timeframe": str(data.loc[i, "timeframe"]),
            "atr": atr_i,
            "atr_norm": atr_i / close_i,
            "volume_ratio": volume_ratio,
            "pivot_high_confirmed": pivot_high_confirmed,
            "pivot_low_confirmed": pivot_low_confirmed,
            "bos_dir": bos_dir,
            "choch_dir": choch_dir,
            "fvg_bullish": fvg_bullish,
            "fvg_bearish": fvg_bearish,
            "fvg_size_atr": fvg_size_atr,
            "nearest_bull_fvg_dist_atr": nearest_bull_fvg_dist,
            "nearest_bear_fvg_dist_atr": nearest_bear_fvg_dist,
            "order_block_bullish": order_block_bullish,
            "order_block_bearish": order_block_bearish,
            "order_block_strength": max_active_ob_strength,
            "nearest_bull_ob_dist_atr": nearest_bull_ob_dist,
            "nearest_bear_ob_dist_atr": nearest_bear_ob_dist,
            "equal_high": equal_high,
            "equal_low": equal_low,
            "liquidity_sweep": liquidity_sweep,
            "premium_discount": premium_discount,
            "price_percentile": price_percentile,
            "trend_strength": trend_strength,
            "distance_from_ema50_atr": distance_from_ema50,
        }
        row.update(session_one_hot(pd.Timestamp(data.loc[i, "time"]).to_pydatetime()))
        feature_rows.append(row)

    return pd.DataFrame(feature_rows, columns=OUTPUT_COLUMNS)
