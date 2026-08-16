"""Market-footprint objectives for feature selection and reward alignment.

SMC/ICT says charts leave a *footprint*: institutional activity shows up as
liquidity sweeps, displacements that leave fair-value gaps, and market-structure
breaks (BOS/CHOCH). A professional trader's edge is "what happens *after* the
footprint prints" — e.g. price reversing after a sweep of resting liquidity.

This module turns that into **causal, fully-vectorised objective labels** used
by the feature selector and (conceptually) by the reward:

* ``structure_r_objective`` — risk-adjusted forward move in the direction of
  market structure (the R-multiple a managed 1:2 trade would aim to capture).
* ``sweep_reversal_objective`` — did price reverse after a liquidity sweep?

Both are computed with pandas/numpy shifting only, so they scale to millions of
bars in memory. Labels are *forward-looking by construction* (that is their
job: they are the outcome to predict) but never leak into the feature set —
the selector only ever compares past features against these future outcomes on
the training slice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(bars: pd.DataFrame, name: str) -> np.ndarray:
    if name in bars.columns:
        return pd.to_numeric(bars[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.zeros(len(bars), dtype=float)


def structure_direction(bars: pd.DataFrame) -> np.ndarray:
    """Return per-bar directional bias in {-1, 0, +1} from ICT structure.

    Prefers a market-structure shift (CHOCH) over a continuation (BOS), and a
    liquidity sweep as confirmation — mirroring how the persona strategy
    assigns long/short scores.
    """
    bos = _numeric(bars, "bos_dir")
    choch = _numeric(bars, "choch_dir")
    sweep = _numeric(bars, "liquidity_sweep")

    direction = np.sign(bos + 2.0 * choch)
    # A sweep against the structural direction flips the bias (the classic ICT
    # reversal: structure is trapped, price reverses after the sweep).
    swept = np.sign(sweep)
    flip = (direction == 0) & (swept != 0)
    direction = np.where(flip, -swept, direction)
    return direction


def structure_r_objective(
    bars: pd.DataFrame,
    *,
    horizon: int = 288,
) -> pd.Series:
    """Forward risk-adjusted return in the structural direction (≈ R-multiple).

    ``(close[i+horizon] - close[i]) / atr[i]`` signed by the structure bias at
    bar ``i``. Continuous, causal, O(n).
    """
    n = len(bars)
    close = _numeric(bars, "close")
    atr = _numeric(bars, "atr")
    direction = structure_direction(bars)

    forward = np.zeros(n, dtype=float)
    if horizon > 0 and n > horizon:
        forward[: n - horizon] = close[horizon:] - close[: n - horizon]
    denom = np.where(atr > 0, atr, np.nan)
    r = forward / denom
    r = np.where(np.isfinite(r), r, 0.0)
    label = direction * r
    return pd.Series(label, index=bars.index, dtype=float)


def sweep_reversal_objective(
    bars: pd.DataFrame,
    *,
    horizon: int = 288,
    atr_mult: float = 1.0,
) -> pd.Series:
    """Binary label: a liquidity sweep that reversed by ≥ ``atr_mult`` × ATR.

    1.0 when price swept one side of liquidity and then moved at least one ATR
    in the opposite direction within ``horizon`` bars; 0.0 otherwise. This is
    the textbook ICT setup: the footprint (sweep) followed by the reversal.
    """
    n = len(bars)
    close = _numeric(bars, "close")
    atr = _numeric(bars, "atr")
    sweep = _numeric(bars, "liquidity_sweep")

    forward = np.zeros(n, dtype=float)
    if horizon > 0 and n > horizon:
        forward[: n - horizon] = close[horizon:] - close[: n - horizon]
    denom = np.where(atr > 0, atr, np.nan)
    move_atr = forward / denom
    move_atr = np.where(np.isfinite(move_atr), move_atr, 0.0)

    reversal = np.where(
        sweep < 0,
        move_atr > atr_mult,  # swept lows, then rallied
        np.where(sweep > 0, move_atr < -atr_mult, False),  # swept highs, then dropped
    )
    return pd.Series(reversal.astype(float), index=bars.index, dtype=float)
