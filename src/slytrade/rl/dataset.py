"""Build scoped RL datasets and environment factories.

The central no-leakage rule: any normalization (e.g. the feature scaler) is
fitted only on the training slice of the data. ``RLDataset`` holds the raw
features; each walk-forward fold fits its own scaler on its train window.

The RL "superbrain" consumes the FULL validated feature set produced by the
pipeline — the ML features PLUS the ICT/SMC features, per-bar tick
microstructure, session flags and the multi-timeframe (htf_*/mtf_bias/
mtf_confluence_score) columns already embedded in the aligned bars. Nothing is
thrown away and nothing unseen is invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.data.alignment import TICK_BAR_FEATURE_COLUMNS
from slytrade.features.ict import FEATURE_COLUMNS as ICT_FEATURE_COLUMNS
from slytrade.intelligence.market_context import MarketContextEngine
from slytrade.intelligence.regime import MarketRegimeEngine
from slytrade.ml.features import ML_FEATURE_COLUMNS, compute_ml_features
from slytrade.rl.environment import RLEnvironmentConfig, SlyTradeRLEnvironment
from slytrade.rl.mode_vector import build_mode_vector

# Columns that are part of the aligned bars but are NOT trading features (they
# are either keys, prices, or already projected into the feature set).
_NON_FEATURE_PREFIXES = (
    "quote_",
)
_NON_FEATURE_COLUMNS = {
    "time",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "decision_time",
    "tick_volume",
    "spread",
    "real_volume",
    "quote_is_fresh",
}


def rl_feature_columns(bars: pd.DataFrame) -> list[str]:
    """Return the ordered feature columns the RL agent should observe.

    ML features are always present. On top of them, every validated feature
    column the aligned bars already carry is adopted: ICT/SMC, per-bar tick
    microstructure, sessions, and the multi-timeframe (htf_*, mtf_bias,
    mtf_confluence_score) columns.
    """
    columns: list[str] = []
    seen: set[str] = set()

    for column in ML_FEATURE_COLUMNS:
        if column in bars.columns and column not in seen:
            columns.append(column)
            seen.add(column)

    for group in (
        ICT_FEATURE_COLUMNS,
        TICK_BAR_FEATURE_COLUMNS,
        ("mtf_bias", "mtf_confluence_score"),
    ):
        for column in group:
            if column in bars.columns and column not in seen:
                columns.append(column)
                seen.add(column)

    for column in bars.columns:
        if column in seen or column in _NON_FEATURE_COLUMNS:
            continue
        if column.startswith(_NON_FEATURE_PREFIXES):
            continue
        if column.startswith(("htf_", "session_")):
            columns.append(column)
            seen.add(column)

    return columns


@dataclass(frozen=True)
class RLDataset:
    """Raw feature + bar matrices for RL training (no fitted scaler yet).

    Each fold fits its own scaler on its own train window to prevent leakage.
    """

    bars: pd.DataFrame
    features: pd.DataFrame
    symbol: str
    point_size: float
    point_value: float

    @property
    def feature_columns(self) -> list[str]:
        return list(self.features.columns)

    def fit_scaler(
        self,
        train_start: int,
        train_end: int,
        *,
        variance_floor: float = 1e-6,
    ) -> dict[str, tuple[float, float]]:
        """Fit per-column (mean, std) on features[train_start:train_end] only."""
        if not 0 <= train_start < train_end <= len(self.features):
            raise ValueError(f"invalid train slice [{train_start}, {train_end}) for {len(self.features)} rows")
        train_features = self.features.iloc[train_start:train_end].copy()
        rng = np.random.default_rng(0)
        params: dict[str, tuple[float, float]] = {}
        for col in train_features.columns:
            series = pd.to_numeric(train_features[col], errors="coerce").fillna(0.0)
            std = float(series.std())
            if std < variance_floor:
                series = series + rng.normal(0.0, variance_floor, size=len(series))
                std = float(series.std())
            params[col] = (float(series.mean()), std if std > 1e-9 else 1.0)
        return params

    def env_factory(
        self,
        start: int,
        end: int,
        *,
        seed: int,
        scaler_params: dict[str, tuple[float, float]],
        config: RLEnvironmentConfig | None = None,
        mode_vector: np.ndarray | None = None,
    ) -> SlyTradeRLEnvironment:
        """Create an environment over bars[start:end] with the given scaler."""
        if not 0 <= start < end <= len(self.bars):
            raise ValueError(f"invalid slice [{start}, {end}) for {len(self.bars)} rows")
        cfg = config or RLEnvironmentConfig(point_size=self.point_size, point_value=self.point_value, seed=seed)
        bars_slice = self.bars.iloc[start:end].reset_index(drop=True)
        features_slice = self.features.iloc[start:end].copy()
        for column, (mean, std) in scaler_params.items():
            if column in features_slice.columns:
                features_slice[column] = (pd.to_numeric(features_slice[column], errors="coerce").fillna(0.0) - mean) / std
        features_slice = features_slice.reset_index(drop=True)
        return SlyTradeRLEnvironment(
            features=features_slice,
            bars=bars_slice,
            config=cfg,
            mode_vector=mode_vector,
        )


def build_rl_dataset(bars: pd.DataFrame) -> RLDataset:
    """Build a raw (unscaled) dataset from validated bars sorted by time.

    The bars must be the pipeline's aligned output: canonical OHLCV plus the
    validated feature columns. ML features are computed on top; the ICT/tick/
    MTF/session features already present in the bars are adopted verbatim.
    """
    if bars.empty:
        raise ValueError("bars frame is empty")
    required = {"time", "symbol", "open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    bars = bars.sort_values("time").reset_index(drop=True)
    ml_features = compute_ml_features(bars)
    if ml_features.empty or len(ml_features) != len(bars):
        raise ValueError("feature computation failed")

    feature_columns = rl_feature_columns(bars)
    adopted = [column for column in feature_columns if column in bars.columns and column not in ml_features.columns]
    features = ml_features.reset_index(drop=True)
    for column in adopted:
        features[column] = pd.to_numeric(bars[column].reset_index(drop=True), errors="coerce")

    features = features.fillna(0.0)
    return RLDataset(
        bars=bars,
        features=features,
        symbol=str(bars["symbol"].iloc[0]),
        point_size=0.01,
        point_value=1.0,
    )


def build_mode_vector_from_bars(
    personality: TraderPersonality,
    bars: pd.DataFrame,
    *,
    index: int,
) -> np.ndarray:
    """Build the mode vector for a specific bar index using history until it.

    Safe for training-time feature generation; the context engine consumes only
    bars up to `index`.
    """
    window = bars.iloc[max(0, index - 120) : index + 1]
    if window.empty:
        window = bars.iloc[0:1]
    context_engine = MarketContextEngine(personality, MarketRegimeEngine())
    context = context_engine.analyze(window)
    return build_mode_vector(personality, context)
