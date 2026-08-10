"""Leakage-safe deterministic feature selection for RL datasets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected: tuple[str, ...]
    scores: dict[str, float]
    train_start: int
    train_end: int


def select_features(
    features: pd.DataFrame,
    close: pd.Series,
    *,
    train_start: int,
    train_end: int,
    max_features: int = 32,
    min_variance: float = 1e-12,
    max_correlation: float = 0.92,
) -> FeatureSelectionResult:
    """Select features using training-only correlation with next-bar returns.

    This is a transparent baseline selector: it is deterministic, handles
    constant columns, and never inspects rows outside the supplied train slice.
    """
    if not 0 <= train_start < train_end <= len(features) or len(close) != len(features):
        raise ValueError("invalid training slice or close alignment")
    if max_features < 1:
        raise ValueError("max_features must be positive")
    if not 0.0 < max_correlation <= 1.0:
        raise ValueError("max_correlation must be in (0, 1]")
    target = pd.to_numeric(close, errors="coerce").pct_change().shift(-1)
    x = features.iloc[train_start:train_end]
    y = target.iloc[train_start:train_end]
    scores: dict[str, float] = {}
    for column in features.columns:
        values = pd.to_numeric(x[column], errors="coerce")
        valid = values.notna() & y.notna()
        if valid.sum() < 3 or float(values[valid].var()) <= min_variance:
            scores[column] = 0.0
            continue
        correlation = float(values[valid].corr(y[valid]))
        scores[column] = abs(correlation) if np.isfinite(correlation) else 0.0
    ordered = sorted(scores, key=lambda name: (-scores[name], name))
    train_values = x.loc[:, ordered].astype(float)
    chosen: list[str] = []
    for column in ordered:
        if len(chosen) >= max_features:
            break
        if not chosen:
            chosen.append(column)
            continue
        correlations = train_values[chosen].corrwith(train_values[column]).abs().fillna(0.0)
        if bool((correlations < max_correlation).all()):
            chosen.append(column)
    selected = tuple(chosen)
    return FeatureSelectionResult(selected, scores, train_start, train_end)


def apply_feature_selection(features: pd.DataFrame, selection: FeatureSelectionResult) -> pd.DataFrame:
    """Apply a previously fitted selection without refitting or leakage."""
    missing = set(selection.selected).difference(features.columns)
    if missing:
        raise ValueError(f"selected features are missing: {sorted(missing)}")
    return features.loc[:, list(selection.selected)].copy()
