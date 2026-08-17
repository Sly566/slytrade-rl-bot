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


# ---------------------------------------------------------------------------
# Dynamic, threshold-free feature selection (production)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicSelectionResult:
    selected: tuple[str, ...]
    scores: dict[str, float]  # best (max) significance score across objectives
    significant: tuple[str, ...]  # features that beat the shadow threshold
    train_start: int
    train_end: int


def select_features_dynamic(
    features: pd.DataFrame,
    objectives: dict[str, pd.Series],
    *,
    train_start: int,
    train_end: int,
    correlation_threshold: float = 0.92,
    n_shadow: int = 5,
    shadow_quantile: float = 1.0,  # 1.0 = max shadow; <1 = that percentile
    seed: int = 42,
) -> DynamicSelectionResult:
    """Select features by significance against trading-aligned objectives.

    Threshold-free (Boruta-style):

    1. For each objective, score every feature by |correlation| with that
       objective, computed **only on the training slice**.
    2. Build ``n_shadow`` shuffled copies of every feature and score them the
       same way. The shadow scores define "what random noise scores."
    3. A feature is *significant* if its score beats the shadow threshold
       (default: the best shadow) on **any** objective.
    4. Greedy correlation de-duplication keeps one feature per redundant
       cluster.

    There is no ``max_features`` — the count emerges from the data and the
    objectives. If nothing beats the shadows, the full feature set is returned
    (no information to prune; never blocks a run).
    """
    if not 0 <= train_start < train_end <= len(features):
        raise ValueError("invalid training slice")
    if not objectives:
        raise ValueError("at least one objective is required")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1]")
    if n_shadow < 1 or not 0.0 < shadow_quantile <= 1.0:
        raise ValueError("invalid shadow configuration")

    x = features.iloc[train_start:train_end]
    n_rows, n_cols = x.shape
    rng = np.random.default_rng(seed)

    def _aligned_objective(series: pd.Series) -> np.ndarray:
        # Objectives are passed ALREADY aligned to the training slice (the
        # caller computes them causally on the train window only), so they are
        # used positionally and only validated for length.
        values = pd.to_numeric(series.reset_index(drop=True), errors="coerce").to_numpy(dtype=float)
        if len(values) != n_rows:
            raise ValueError(
                f"objective length {len(values)} does not match train slice length {n_rows}; "
                "objectives must be pre-aligned to the train window"
            )
        return values

    objective_arrays = [np.asarray(_aligned_objective(series), dtype=float) for series in objectives.values()]

    # One float32 matrix for the train slice (half the float64 footprint), plus
    # one reusable shuffled buffer. Scores are reduced in float64 per column so
    # the ranking matches the old float64 math, but never by materialising a
    # whole float64 frame (or five shuffled copies of it).
    X = x.to_numpy(dtype=np.float32)

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt((a @ a) * (b @ b)))
        if denom <= 1e-15:
            return 0.0
        return float((a @ b) / denom)

    def _score_columns(mat: np.ndarray) -> np.ndarray:
        """Return an [n_cols, n_objectives] matrix of |pearson| scores."""
        out = np.zeros((mat.shape[1], len(objective_arrays)), dtype=float)
        for oi, target in enumerate(objective_arrays):
            finite = np.isfinite(target)
            if finite.sum() < 3:
                continue
            for ci in range(mat.shape[1]):
                valid = finite & np.isfinite(mat[:, ci])
                if valid.sum() < 3:
                    continue
                column = mat[:, ci][valid].astype(np.float64)
                if float(column.var()) <= 1e-12:
                    continue
                correlation = _pearson(column, target[valid])
                out[ci, oi] = abs(correlation) if np.isfinite(correlation) else 0.0
        return out

    real_scores = _score_columns(X)

    # Shadow scores: for each objective, threshold = quantile over all shadows.
    # One shuffled buffer is reused across shadows (never 5 simultaneous copies).
    shadow_thresholds = np.zeros(len(objective_arrays), dtype=float)
    shuffled = np.empty_like(X)
    for oi in range(len(objective_arrays)):
        shadow_scores: list[float] = []
        for _ in range(n_shadow):
            for ci in range(n_cols):
                shuffled[:, ci] = rng.permutation(X[:, ci])
            shadow_scores.extend(_score_columns(shuffled)[:, oi].tolist())
        shadow_thresholds[oi] = float(np.quantile(shadow_scores, shadow_quantile))

    significant: list[str] = []
    best_scores: dict[str, float] = {}
    for ci, column in enumerate(features.columns):
        best = float(real_scores[ci].max())
        beats = bool((real_scores[ci] > shadow_thresholds).any())
        best_scores[column] = best
        if beats:
            significant.append(column)

    if not significant:
        return DynamicSelectionResult(
            selected=tuple(features.columns),
            scores=best_scores,
            significant=(),
            train_start=train_start,
            train_end=train_end,
        )

    ordered = sorted(significant, key=lambda name: (-best_scores[name], name))
    train_values = x.loc[:, ordered]
    chosen: list[str] = []
    for column in ordered:
        if not chosen:
            chosen.append(column)
            continue
        correlations = train_values[chosen].corrwith(train_values[column]).abs().fillna(0.0)
        if bool((correlations < correlation_threshold).all()):
            chosen.append(column)
    return DynamicSelectionResult(
        selected=tuple(chosen),
        scores=best_scores,
        significant=tuple(significant),
        train_start=train_start,
        train_end=train_end,
    )


def apply_dynamic_selection(features: pd.DataFrame, selection: DynamicSelectionResult) -> pd.DataFrame:
    """Apply a previously fitted dynamic selection (column filter)."""
    missing = set(selection.selected).difference(features.columns)
    if missing:
        raise ValueError(f"selected features are missing: {sorted(missing)}")
    return features.loc[:, list(selection.selected)].copy()
