"""Build scoped RL datasets and environment factories.

The central no-leakage rule: any normalization (e.g. the feature scaler) is
fitted only on the training slice of the data. `RLDataset` holds the raw
features; each walk-forward fold fits its own scaler on its train window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.intelligence.market_context import MarketContextEngine
from slytrade.intelligence.regime import MarketRegimeEngine
from slytrade.ml.features import apply_scaler, compute_ml_features, fit_scaler
from slytrade.rl.environment import RLEnvironmentConfig, SlyTradeRLEnvironment
from slytrade.rl.mode_vector import build_mode_vector


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
        for col in train_features.columns:
            if float(train_features[col].std()) < variance_floor:
                train_features[col] = train_features[col] + rng.normal(
                    0.0, variance_floor, size=len(train_features)
                )
        return fit_scaler(train_features)

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
        features_slice = apply_scaler(self.features.iloc[start:end], scaler_params).reset_index(drop=True)
        return SlyTradeRLEnvironment(
            features=features_slice,
            bars=bars_slice,
            config=cfg,
            mode_vector=mode_vector,
        )


def build_rl_dataset(bars: pd.DataFrame) -> RLDataset:
    """Build a raw (unscaled) dataset from canonical bars sorted by time."""
    if bars.empty:
        raise ValueError("bars frame is empty")
    required = {"time", "symbol", "open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    bars = bars.sort_values("time").reset_index(drop=True)
    features = compute_ml_features(bars)
    if features.empty or len(features) != len(bars):
        raise ValueError("feature computation failed")

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
