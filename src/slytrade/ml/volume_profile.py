"""Causal rolling volume-profile features for bar and tick-volume data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volume_profile_features(
    bars: pd.DataFrame,
    *,
    window: int = 60,
    bins: int = 24,
) -> pd.DataFrame:
    """Return rolling POC/value-area features using data available per bar.

    The profile uses typical price and tick volume. Each row's profile includes
    only the current and previous ``window - 1`` rows, so it is safe for
    decision-time feature generation.
    """
    if window < 2 or bins < 4:
        raise ValueError("window must be >= 2 and bins must be >= 4")
    required = {"high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")
    if bars.empty:
        return pd.DataFrame(index=bars.index)

    high = pd.to_numeric(bars["high"], errors="coerce").ffill().bfill()
    low = pd.to_numeric(bars["low"], errors="coerce").ffill().bfill()
    close = pd.to_numeric(bars["close"], errors="coerce").ffill().bfill()
    volume = (
        pd.to_numeric(bars["tick_volume"], errors="coerce").fillna(0.0)
        if "tick_volume" in bars.columns
        else pd.Series(1.0, index=bars.index)
    )
    typical = (high + low + close) / 3.0
    rows: list[dict[str, float]] = []
    for end in range(len(bars)):
        start = max(0, end - window + 1)
        prices = typical.iloc[start : end + 1].to_numpy(dtype=float)
        weights = volume.iloc[start : end + 1].to_numpy(dtype=float)
        lo = float(np.min(low.iloc[start : end + 1]))
        hi = float(np.max(high.iloc[start : end + 1]))
        width = max((hi - lo) / bins, 1e-12)
        indices = np.clip(((prices - lo) / width).astype(int), 0, bins - 1)
        histogram = np.bincount(indices, weights=np.maximum(weights, 0.0), minlength=bins)
        total = float(histogram.sum())
        if total <= 0:
            histogram = np.bincount(indices, minlength=bins).astype(float)
            total = float(histogram.sum())
        poc_bin = int(np.argmax(histogram))
        target = total * 0.70
        left = right = poc_bin
        covered = float(histogram[poc_bin])
        while covered < target and (left > 0 or right < bins - 1):
            left_value = histogram[left - 1] if left > 0 else -1.0
            right_value = histogram[right + 1] if right < bins - 1 else -1.0
            if right_value >= left_value and right < bins - 1:
                right += 1
                covered += float(histogram[right])
            elif left > 0:
                left -= 1
                covered += float(histogram[left])
            else:
                break
        poc = lo + (poc_bin + 0.5) * width
        value_low = lo + left * width
        value_high = lo + (right + 1) * width
        rows.append(
            {
                "vp_poc": poc,
                "vp_value_area_low": value_low,
                "vp_value_area_high": value_high,
                "vp_position": float((close.iloc[end] - value_low) / max(value_high - value_low, width)),
                "vp_poc_distance_atr": 0.0,
                "vp_volume_concentration": float(histogram[poc_bin] / max(total, 1e-12)),
            }
        )
    output = pd.DataFrame(rows, index=bars.index)
    atr_proxy = (high - low).rolling(min(window, 14), min_periods=1).mean().replace(0.0, np.nan)
    output["vp_poc_distance_atr"] = (close - output["vp_poc"]).abs() / atr_proxy
    return output.replace([np.inf, -np.inf], 0.0).fillna(0.0)
