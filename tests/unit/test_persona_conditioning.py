"""The persona + regime conditioning channel for the RL observation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.rl.mode_vector import (
    PERSONA_TRAIT_NAMES,
    build_mode_matrix,
    build_mode_vector,
    mode_matrix_columns,
    persona_fingerprint,
)


def test_persona_fingerprint_length_and_values() -> None:
    persona = TraderPersonality(aggression=0.8, selectivity=0.9)
    fingerprint = persona_fingerprint(persona)
    assert len(fingerprint) == len(PERSONA_TRAIT_NAMES)
    assert fingerprint[0] == 0.8  # aggression is first
    assert fingerprint[1] == 0.9  # selectivity is second
    assert (fingerprint >= 0.0).all() and (fingerprint <= 1.0).all()


def test_build_mode_vector_now_includes_persona() -> None:
    """Regression: build_mode_vector used to ignore `personality` entirely."""
    persona = TraderPersonality()
    context = {
        "volatility": "high",
        "trend": "bull",
        "session": "london",
        "regime_score": 0.8,
        "premium_discount": -0.1,
        "mtf_bias": 1.0,
    }
    vector = build_mode_vector(persona, context)
    # 3 vol + 3 trend + 6 session + 3 scalars + 18 persona traits.
    assert len(vector) == 3 + 3 + 6 + 3 + len(PERSONA_TRAIT_NAMES)
    # A different persona changes the tail of the vector.
    other = TraderPersonality(aggression=1.0, selectivity=0.1)
    vector_other = build_mode_vector(other, context)
    assert not np.allclose(vector, vector_other)


def _bars(n: int = 200, *, rising: bool = True) -> pd.DataFrame:
    times = pd.date_range("2026-08-14T08:00:00", periods=n, freq="min", tz="UTC")
    close = 100.0 + (pd.Series(range(n), dtype=float) * 0.01 if rising else -pd.Series(range(n), dtype=float) * 0.01)
    atr = 0.5 + pd.Series(np.random.default_rng(0).normal(0, 0.2, n)).abs()
    return pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "atr_norm": atr / close,
            "trend_strength": pd.Series(0.3 if rising else -0.3, index=range(n)),
            "premium_discount": 0.0,
            "mtf_bias": 1.0 if rising else -1.0,
        }
    )


def test_build_mode_matrix_shape_and_columns() -> None:
    bars = _bars()
    matrix = build_mode_matrix(bars, TraderPersonality())
    assert list(matrix.columns) == mode_matrix_columns()
    assert len(matrix) == len(bars)
    # Regime score is within [0, 1].
    assert matrix["mode_regime_score"].between(0.0, 1.0).all()
    # Persona columns are constant across the dataset (fixed persona).
    assert matrix["mode_p_aggression"].nunique() == 1


def test_build_mode_matrix_captures_trend_regime() -> None:
    bull = build_mode_matrix(_bars(200, rising=True), TraderPersonality())
    bear = build_mode_matrix(_bars(200, rising=False), TraderPersonality())
    # A bull market should spend most time in the bull regime class.
    assert (bull["mode_trend_bull"] == 1.0).mean() > 0.9
    assert (bear["mode_trend_bear"] == 1.0).mean() > 0.9


def test_dataset_includes_mode_columns() -> None:
    from slytrade.rl.dataset import build_rl_dataset

    bars = _bars(150)
    dataset = build_rl_dataset(bars, TraderPersonality())
    for column in mode_matrix_columns():
        assert column in dataset.features.columns, column
    # Feature count = ML + adopted ICT + mode columns.
    assert len(dataset.features.columns) >= len(mode_matrix_columns())
