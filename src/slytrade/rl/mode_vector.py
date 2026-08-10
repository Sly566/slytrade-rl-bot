"""Build the global "mode vector" that appends macro/personality state to the
RL observation. This vector is the personality-conditioning channel: it lets a
single policy behave differently depending on the market regime and the
configured trader persona.
"""
from __future__ import annotations

import numpy as np

from slytrade.config.trader_personality import TraderPersonality

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


def build_mode_vector(personality: TraderPersonality, context: dict) -> np.ndarray:
    """Return a fixed-size vector of global context + persona traits.

    Order (NUM_MODE_COLUMNS = 6):
    [volatility one-hot idx, trend one-hot idx, session one-hot idx,
     regime_score, premium_discount, mtf_bias]
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

    regime_score = float(context.get("regime_score", 0.5))
    premium = float(context.get("premium_discount", 0.0))
    mtf_bias = float(context.get("mtf_bias", 0.0))

    return np.concatenate(
        [vol, tr, ses, np.array([regime_score, premium, mtf_bias], dtype=np.float32)]
    )
