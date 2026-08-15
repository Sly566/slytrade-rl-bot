"""Persona + market-regime conditioning for the RL observation.

The RL "superbrain" is a single policy that must behave like the trader persona
it is trained for. This module builds the conditioning channel:

* **persona fingerprint** — the trader's traits (aggression, selectivity,
  patience, discipline, liquidity focus, structure focus, …) as a fixed
  [0,1] vector. It makes the persona part of the observation, so the policy
  knows *who it is* and can specialise accordingly.
* **market regime** — per-bar volatility/trend classification plus the regime
  quality score, computed causally (only bars ≤ the current one), so the
  policy can adapt its aggression/selectivity to the current market state.

Both are appended to the observation as ``mode_*`` columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality

# Ordered persona traits exposed to the RL (all in [0, 1]).
PERSONA_TRAIT_NAMES: tuple[str, ...] = (
    "aggression",
    "selectivity",
    "risk_tolerance",
    "scalping_bias",
    "day_trading_bias",
    "macro_respect",
    "session_sensitivity",
    "conviction",
    "patience",
    "discipline",
    "adaptability",
    "time_pressure",
    "structure_focus",
    "liquidity_focus",
    "trade_duration_bias",
    "cut_losses_fast",
    "let_winners_run",
    "edge_optimism",
)

VOLATILITY_INDEX: dict[str, int] = {"low": 0, "normal": 1, "high": 2}
TREND_INDEX: dict[str, int] = {"bear": 0, "ranging": 1, "bull": 2}
SESSION_INDEX: dict[str, int] = {
    "asia": 0,
    "london": 1,
    "ny_am": 2,
    "ny_pm": 3,
    "other": 4,
    "unknown": 5,
}

# Regime quality weights (mirrors slytrade.intelligence.regime).
_REGIME_VOL_WEIGHT = {0: 0.7, 1: 1.0, 2: 0.85}  # low / normal / high


def persona_fingerprint(personality: TraderPersonality) -> np.ndarray:
    """Return the persona's trait vector (length == len(PERSONA_TRAIT_NAMES))."""
    return np.asarray(
        [float(getattr(personality, name, 0.5)) for name in PERSONA_TRAIT_NAMES],
        dtype=np.float32,
    )


def mode_matrix_columns() -> list[str]:
    """Column names produced by :func:`build_mode_matrix`."""
    volatility = [f"mode_vol_{name}" for name in ("low", "normal", "high")]
    trend = [f"mode_trend_{name}" for name in ("bear", "ranging", "bull")]
    persona = [f"mode_p_{name}" for name in PERSONA_TRAIT_NAMES]
    return volatility + trend + ["mode_regime_score"] + persona


def build_mode_vector(personality: TraderPersonality, context: dict) -> np.ndarray:
    """Return a fixed-size vector of market context + persona traits.

    Order: volatility one-hot (3), trend one-hot (3), session one-hot (6),
    [regime_score, premium_discount, mtf_bias] (3), persona traits (18).
    """
    volatility = context.get("volatility", "normal")
    trend = context.get("trend", "ranging")
    session = context.get("session", "unknown")

    vol = np.zeros(3, dtype=np.float32)
    vol[VOLATILITY_INDEX.get(volatility, 1)] = 1.0

    tr = np.zeros(3, dtype=np.float32)
    tr[TREND_INDEX.get(trend, 1)] = 1.0

    ses = np.zeros(6, dtype=np.float32)
    ses[SESSION_INDEX.get(session, 5)] = 1.0

    scalars = np.asarray(
        [
            float(context.get("regime_score", 0.5)),
            float(context.get("premium_discount", 0.0)),
            float(context.get("mtf_bias", 0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([vol, tr, ses, scalars, persona_fingerprint(personality)])


def build_mode_matrix(
    bars: pd.DataFrame,
    personality: TraderPersonality,
    *,
    volatility_lookback: int = 100,
    volatile_z_threshold: float = 0.8,
    quiet_z_threshold: float = -0.8,
    trend_threshold_atr: float = 0.15,
) -> pd.DataFrame:
    """Build the per-bar causal mode matrix (regime + persona).

    Vectorised (no per-bar Python loop) and strictly causal: the volatility
    classification at bar ``i`` uses a rolling window ending at ``i``; the trend
    and session use the bar's own already-computed features. Returns a DataFrame
    indexed like ``bars`` with :func:`mode_matrix_columns`.
    """
    n = len(bars)
    columns = mode_matrix_columns()
    if n == 0:
        return pd.DataFrame(columns=columns)

    def _numeric(name: str) -> pd.Series:
        if name in bars.columns:
            return pd.to_numeric(bars[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=bars.index, dtype=float)

    # --- volatility regime (rolling z-score of ATR-normalised range) ---------
    atr_norm = _numeric("atr_norm")
    roll = atr_norm.rolling(volatility_lookback, min_periods=20)
    z = (atr_norm - roll.mean()) / roll.std().replace(0.0, np.nan)
    z = z.fillna(0.0).to_numpy(dtype=float)
    vol_idx = np.where(z > volatile_z_threshold, 2, np.where(z < quiet_z_threshold, 0, 1))

    # --- trend regime --------------------------------------------------------
    trend_strength = _numeric("trend_strength").to_numpy(dtype=float)
    trend_idx = np.where(
        trend_strength > trend_threshold_atr,
        2,
        np.where(trend_strength < -trend_threshold_atr, 0, 1),
    )

    # --- session (for the regime quality score) ------------------------------
    times = pd.to_datetime(bars["time"], utc=True)
    hours = times.dt.hour.to_numpy(dtype=int)
    session_idx = np.select(
        [hours < 7, hours < 12, hours < 16, hours < 21],
        [0, 1, 2, 3],  # asia / london / ny_am / ny_pm
        default=4,  # other
    )
    session_score = np.where(np.isin(session_idx, [1, 2, 3]), 1.0, np.where(session_idx == 0, 0.6, 0.4))

    # --- regime quality score -------------------------------------------------
    vol_weight = np.asarray([_REGIME_VOL_WEIGHT[int(v)] for v in vol_idx], dtype=float)
    trend_weight = np.where(trend_idx != 1, 0.75, 0.45)
    regime_score = np.minimum(0.35 * vol_weight + 0.35 * trend_weight + 0.30 * session_score, 1.0)

    # --- assemble -------------------------------------------------------------
    vol_onehot = np.eye(3, dtype=np.float32)[vol_idx]
    trend_onehot = np.eye(3, dtype=np.float32)[trend_idx]
    persona = np.tile(persona_fingerprint(personality), (n, 1)).astype(np.float32)
    matrix = np.concatenate(
        [vol_onehot, trend_onehot, regime_score.astype(np.float32)[:, None], persona],
        axis=1,
    )
    return pd.DataFrame(matrix, index=bars.index, columns=columns)
