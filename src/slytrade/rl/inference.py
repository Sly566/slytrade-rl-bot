"""Inference-time strategy for a trained RL policy.

``RLPolicyStrategy`` adapts the SlyTrade RL environment's observation/action
convention into the ``BarStrategy.on_bar`` interface used by every backtest
engine and the paper/demo runtime loops. This is the missing link that turns a
saved artifact into something that can actually trade (on paper first).

Causality is preserved: features are recomputed from a rolling window of bars
strictly up to the current bar, scaled with the artifact's fitted scaler, and
the policy only ever sees ``bars <= index``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from slytrade.execution.models import OrderIntent, Side
from slytrade.ml.features import compute_ml_features
from slytrade.risk.sizing import risk_based_volume


@dataclass
class RLPolicyStrategy:
    """Emit OrderIntents from a trained policy using causal rolling features.

    Actions follow the environment convention: 0 hold, 1 long, 2 short,
    3 flatten. Entries are sized on a risk budget using the bar ATR as stop
    distance; flattens close the whole position.
    """

    model: Any
    feature_columns: tuple[str, ...]
    scaler_params: dict[str, tuple[float, float]]
    symbol: str = "XAUUSD"
    mode_vector: np.ndarray | None = None
    history_window: int = 250
    risk_per_trade: float = 0.005
    point_value: float = 1.0
    equity: float = 100_000.0
    _history: list[pd.Series] = field(default_factory=list, init=False)
    _side: str = field(default="flat", init=False)

    # -- internal -----------------------------------------------------------
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)

    def _observation(self, bar: pd.Series) -> np.ndarray:
        frame = self._frame()
        features = compute_ml_features(frame)
        row = features.iloc[-1].copy()
        vector = np.asarray([float(row.get(column, 0.0)) for column in self.feature_columns], dtype=np.float32)
        for i, column in enumerate(self.feature_columns):
            if column in self.scaler_params:
                mean, std = self.scaler_params[column]
                vector[i] = (vector[i] - mean) / (std if std > 1e-9 else 1.0)
        if self.mode_vector is not None:
            vector = np.concatenate((vector, np.asarray(self.mode_vector, dtype=np.float32)))
        return vector

    def _predict(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def _sized_volume(self, bar: pd.Series) -> float:
        atr = float(bar.get("atr", 0.0) or 0.0)
        stop_distance = max(atr, 0.10)
        volume = risk_based_volume(
            self.equity,
            stop_distance,
            risk_per_trade=self.risk_per_trade,
            point_value=self.point_value,
            volume_max=100.0,
        )
        return volume if volume > 0 else 0.01

    # -- BarStrategy interface ----------------------------------------------
    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        self._history.append(bar.copy())
        if len(self._history) > self.history_window:
            self._history.pop(0)

        if len(self._history) < 20:  # need a minimal window for stable features
            return None

        observation = self._observation(bar)
        action = self._predict(observation)

        if action == 3 and self._side != "flat":
            side = Side.SELL if self._side == "long" else Side.BUY
            self._side = "flat"
            return OrderIntent(symbol=self.symbol, side=side, volume=1.0, reason="rl_flatten")
        if action == 1 and self._side != "long":
            self._side = "long"
            return OrderIntent(symbol=self.symbol, side=Side.BUY, volume=self._sized_volume(bar), reason="rl_long")
        if action == 2 and self._side != "short":
            self._side = "short"
            return OrderIntent(symbol=self.symbol, side=Side.SELL, volume=self._sized_volume(bar), reason="rl_short")
        return None

    def reset(self) -> None:
        self._history.clear()
        self._side = "flat"


def strategy_from_artifact(model, artifact, *, symbol: str | None = None) -> RLPolicyStrategy:
    """Build an inference strategy from ``(model, artifact)`` (see
    :func:`slytrade.rl.deployment.load_model_artifact`)."""
    meta = artifact.meta
    return RLPolicyStrategy(
        model=model,
        feature_columns=tuple(meta.feature_columns),
        scaler_params=artifact.scaler_params,
        symbol=symbol or meta.symbol,
    )
