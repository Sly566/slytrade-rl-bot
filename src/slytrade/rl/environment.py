from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from slytrade.execution.ledger import TradeLedger
from slytrade.execution.models import OrderIntent, Side

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


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


@dataclass(frozen=True)
class RLEnvironmentConfig:
    initial_balance: float = 100_000.0
    point_size: float = 0.01
    point_value: float = 1.0
    transaction_cost: float = 0.0002
    max_position_volume: float = 10.0
    risk_per_trade: float = 0.005
    seed: int = 42
    # "raw" = plain equity delta; "risk_adjusted" = drawdown/turnover-penalised
    # reward from slytrade.rl.rewards (recommended for production training).
    reward_type: str = "raw"
    drawdown_tolerance: float = 0.05


if gym is not None:

    class SlyTradeRLEnvironment(gym.Env):
        """Causal feature environment with bounded target-position actions."""

        metadata: dict[str, object] = {"render_modes": []}

        def __init__(
            self,
            features: pd.DataFrame,
            bars: pd.DataFrame,
            config: RLEnvironmentConfig | None = None,
            mode_vector: np.ndarray | None = None,
            ledger: TradeLedger | None = None,
            **_: object,
        ):
            if features.empty or len(features) != len(bars):
                raise ValueError("features and bars must be non-empty and aligned")
            required = {"time", "symbol", "open", "high", "low", "close"}
            if not required.issubset(bars.columns):
                raise ValueError(f"bars missing required columns: {sorted(required.difference(bars.columns))}")
            super().__init__()
            self.features = features.reset_index(drop=True)
            self.bars = bars.reset_index(drop=True)
            self.config = config or RLEnvironmentConfig()
            self.mode_vector = mode_vector
            shape = len(self.features.columns) + (len(mode_vector) if mode_vector is not None else 0)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(shape,), dtype=np.float32)
            self.action_space = spaces.Discrete(4)
            self.ledger = ledger or TradeLedger()
            self.current_step = 0
            self._position = 0
            self._equity = self.config.initial_balance
            self._peak_equity = self.config.initial_balance
            self._entry_price = 0.0

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self.current_step = 0
            self._position = 0
            self._equity = self.config.initial_balance
            self._peak_equity = self.config.initial_balance
            self._entry_price = 0.0
            self.ledger.records.clear()
            return self._observation(), {}

        def step(self, action: int):
            if action not in (0, 1, 2, 3):
                raise ValueError("action must be 0 (hold), 1 (long), 2 (short), or 3 (flatten)")
            if self.current_step >= len(self.bars):
                raise RuntimeError("step() called past the end of the episode; call reset()")
            previous = float(self.bars.iloc[self.current_step]["close"])
            old_position = self._position
            target = self._position
            if action == 1:
                target = 1
            elif action == 2:
                target = -1
            elif action == 3:
                target = 0
            turnover = abs(target - old_position)
            if turnover and target:
                intent = OrderIntent(
                    symbol=str(self.bars.iloc[self.current_step]["symbol"]),
                    side=Side.BUY if target > 0 else Side.SELL,
                    volume=min(self.config.max_position_volume, self.config.risk_per_trade),
                    reason="rl_entry",
                )
                self.ledger.record_fill(
                    intent,
                    volume=intent.volume,
                    price=previous,
                    commission=self.config.transaction_cost * intent.volume,
                    realized_pnl=0.0,
                    event_time=pd.Timestamp(self.bars.iloc[self.current_step]["time"]).to_pydatetime(),
                )
            self.current_step += 1
            current = float(self.bars.iloc[min(self.current_step, len(self.bars) - 1)]["close"])
            previous_equity = self._equity
            equity_delta = previous_equity * (target * (current - previous) / max(previous, 1e-12))
            equity_delta -= previous_equity * turnover * self.config.transaction_cost
            self._equity = previous_equity + equity_delta
            self._peak_equity = max(self._peak_equity, self._equity)
            self._position = target
            terminated = self.current_step >= len(self.bars) - 1 or self._equity <= 0

            if self.config.reward_type == "risk_adjusted":
                from slytrade.rl.rewards import RewardConfig, shaped_reward

                # The raw delta already includes transaction costs, so the shaper's
                # cost term is disabled to avoid double counting.
                reward = shaped_reward(
                    previous_equity=previous_equity,
                    equity=self._equity,
                    position=old_position,
                    target_position=target,
                    peak_equity=self._peak_equity,
                    config=RewardConfig(
                        transaction_cost=0.0,
                        drawdown_tolerance=self.config.drawdown_tolerance,
                    ),
                )
            elif self.config.reward_type == "trade_pnl":
                # Sparse, trade-close reward: realized PnL when a position closes,
                # zero while holding. This trains trade quality (win/loss size)
                # instead of bar-to-bar mark-to-market noise, and stops the
                # policy from churning.
                reward = self._trade_pnl_reward(old_position, target, turnover, previous)
            else:
                reward = equity_delta

            return self._observation(), float(reward), terminated, False, {
                "equity": float(self._equity),
                "n_trades": len(self.ledger.records),
            }

        def _trade_pnl_reward(self, old_position: int, target: int, turnover: int, price: float) -> float:
            """Realized-PnL reward, paid only when exposure changes.

            Opening a position costs transaction costs (small negative reward);
            closing (or reversing) realizes PnL from the entry price to the exit
            price. Holding (or staying flat) yields exactly zero reward.
            """
            cost = turnover * self.config.transaction_cost

            if old_position == 0 and target != 0:
                # Opening a new position: entry price is the decision price.
                self._entry_price = price
                return -cost

            if old_position != 0 and target != old_position:
                # Closing (target == 0) or reversing: realize the closed leg.
                if self._entry_price > 0:
                    realized = old_position * (price - self._entry_price) / self._entry_price
                else:
                    realized = 0.0
                self._entry_price = price if target != 0 else 0.0
                return realized - cost

            # Holding an open position or staying flat: no reward.
            return 0.0

        def _observation(self) -> np.ndarray:
            index = min(self.current_step, len(self.features) - 1)
            row = self.features.iloc[index].to_numpy(dtype=np.float32)
            if self.mode_vector is not None:
                row = np.concatenate((row, np.asarray(self.mode_vector, dtype=np.float32)))
            return np.asarray(row, dtype=np.float32)

else:

    class SlyTradeRLEnvironment:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError("SlyTradeRLEnvironment requires the optional 'rl' dependencies")
