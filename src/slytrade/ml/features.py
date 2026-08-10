"""Causal technical feature stack for ML/RL.

This module intentionally returns features that are *causal* (rolling windows
only, never centered) so the RL environment cannot peek into the future. The
computation mirrors the spirit of the ICT/SMC feature engine but is compact
and vectorized for training speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ML_FEATURE_COLUMNS = [
    "ml_ret_1",
    "ml_ret_5",
    "ml_ret_20",
    "ml_volatility_20",
    "ml_atr_norm",
    "ml_ema_fast",
    "ml_ema_slow",
    "ml_ema_cross",
    "ml_rsi_14",
    "ml_volume_ratio",
    "ml_high_low_range",
    "ml_body_ratio",
]

ML_SCALE_COLUMNS = ML_FEATURE_COLUMNS[:-2]  # scaled features (range/body dropped)


def compute_ml_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute causal ML features on canonical OHLCV bars.

    Required columns: time, open, high, low, close (tick_volume optional).
    Returns a DataFrame indexed like `bars` with the ML_FEATURE_COLUMNS.
    """
    required = {"open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    if bars.empty:
        return pd.DataFrame(columns=ML_FEATURE_COLUMNS, index=bars.index)

    data = bars.sort_values("time").reset_index(drop=True).copy()
    open_ = pd.to_numeric(data["open"], errors="coerce").ffill().fillna(0.0)
    high = pd.to_numeric(data["high"], errors="coerce").ffill().fillna(0.0)
    low = pd.to_numeric(data["low"], errors="coerce").ffill().fillna(0.0)
    close = pd.to_numeric(data["close"], errors="coerce").ffill().fillna(0.0)

    volume = (
        pd.to_numeric(data["tick_volume"], errors="coerce").fillna(0.0)
        if "tick_volume" in data.columns
        else pd.Series(1.0, index=data.index)
    )

    out = pd.DataFrame(index=data.index)
    out["ml_ret_1"] = close.pct_change(1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["ml_ret_5"] = close.pct_change(5).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["ml_ret_20"] = close.pct_change(20).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out["ml_volatility_20"] = out["ml_ret_1"].rolling(20).std().fillna(0.0)

    # ATR-like normalized range
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14.0, min_periods=1).mean()
    out["ml_atr_norm"] = (atr / close.replace(0.0, np.nan)).fillna(0.0)

    # EMAs
    out["ml_ema_fast"] = close.ewm(span=10, adjust=False).mean()
    out["ml_ema_slow"] = close.ewm(span=50, adjust=False).mean()
    out["ml_ema_cross"] = ((out["ml_ema_fast"] - out["ml_ema_slow"]) / close.replace(0.0, np.nan)).fillna(0.0)

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, min_periods=1).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["ml_rsi_14"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0) / 100.0  # normalize to [0,1]

    # Volume ratio
    vol_sma = volume.rolling(20, min_periods=1).mean().replace(0.0, np.nan)
    out["ml_volume_ratio"] = (volume / vol_sma).fillna(1.0).clip(0.0, 10.0)

    # Range and body (kept unscaled for scale fitting)
    out["ml_high_low_range"] = ((high - low) / close.replace(0.0, np.nan)).fillna(0.0)
    body = (close - open_).abs()
    out["ml_body_ratio"] = (body / (high - low).replace(0.0, np.nan)).fillna(0.0)

    # Defensive clip on ratio-type features
    for col in ["ml_volume_ratio", "ml_high_low_range", "ml_body_ratio"]:
        out[col] = out[col].clip(0.0, 10.0)

    out = out.reindex(index=bars.index)
    out = out[ML_FEATURE_COLUMNS]
    return out


def fit_scaler(features: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Fit a per-column (mean, std) from features. Mean/std of 0 columns are kept as 0/1."""
    params: dict[str, tuple[float, float]] = {}
    for col in ML_SCALE_COLUMNS:
        mean = float(features[col].mean()) if col in features.columns else 0.0
        std = float(features[col].std()) if col in features.columns else 0.0
        if std <= 1e-9:
            std = 1.0
        params[col] = (mean, std)
    return params


def apply_scaler(features: pd.DataFrame, params: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Z-score features using pre-fitted params (no lookahead)."""
    out = features.copy()
    for col, (mean, std) in params.items():
        if col not in out.columns:
            continue
        out[col] = (out[col] - mean) / std
    return out
