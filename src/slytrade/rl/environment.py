from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

gym: Any
spaces: Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


@dataclass(frozen=True)
class EnvironmentConfig:
    initial_balance: float = 100_000.0
    transaction_cost: float = 0.0002


if gym is not None:

    class TradingEnvironment(gym.Env):
        """Long/short/flat environment with explicit costs and bounded exposure."""

        metadata: dict[str, object] = {"render_modes": []}

        def __init__(self, bars: pd.DataFrame, config: EnvironmentConfig | None = None):
            super().__init__()
            if "close" not in bars.columns or len(bars) < 2:
                raise ValueError("bars must contain at least two close prices")
            self.bars = bars.reset_index(drop=True)
            self.config = config or EnvironmentConfig()
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
            self._index = 0
            self._position = 0
            self._equity = self.config.initial_balance

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self._index = 0
            self._position = 0
            self._equity = self.config.initial_balance
            return self._observation(), {}

        def step(self, action: int):
            if action not in (0, 1, 2):
                raise ValueError("action must be 0 (short), 1 (flat), or 2 (long)")
            target = action - 1
            previous_close = float(self.bars.iloc[self._index]["close"])
            self._index += 1
            current_close = float(self.bars.iloc[self._index]["close"])
            price_return = (current_close - previous_close) / max(previous_close, 1e-12)
            turnover = abs(target - self._position)
            reward = float(self._equity * (target * price_return - turnover * self.config.transaction_cost))
            self._equity += reward
            self._position = target
            terminated = self._index >= len(self.bars) - 1 or self._equity <= 0
            return self._observation(), reward, terminated, False, {"equity": self._equity}

        def _observation(self) -> np.ndarray:
            row = self.bars.iloc[self._index]
            close = float(row["close"])
            previous = float(self.bars.iloc[max(0, self._index - 1)]["close"])
            change = (close - previous) / max(previous, 1e-12)
            return np.asarray(
                [close, change, float(self._position), self._equity / self.config.initial_balance],
                dtype=np.float32,
            )

else:

    class TradingEnvironment:  # type: ignore[no-redef]
        def __init__(self, bars: pd.DataFrame, config: EnvironmentConfig | None = None):
            raise ImportError("TradingEnvironment requires the optional 'rl' dependencies")
