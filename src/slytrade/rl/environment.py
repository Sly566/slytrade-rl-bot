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
    # Episodes are truncated after this many bars (0 = run to the end of the
    # slice). Without a bound, a 1y M1 dataset is a single 350k-step episode,
    # which is why the first models churned tens of thousands of trades and
    # blew up the account before ever seeing a reward signal resolve.
    episode_length_bars: int = 1000
    # Route RL entries through the SAME managed-exit engine as the backtests:
    # ATR stop-loss / take-profit (and optional trailing). With this on, the
    # agent's reward is realized PnL at real exits instead of mark-to-market
    # noise, so it learns trade management exactly like the persona strategy.
    use_managed_exits: bool = True
    stop_loss_atr: float = 1.0
    take_profit_atr: float = 2.0
    trailing_stop_atr: float | None = None
    # Production reward (reward_type="r_multiple") — the unit is R, the risk
    # at entry. Entry/regret shaping is driven by the ICT footprint score.
    setup_score_threshold: int = 4
    entry_quality_bonus: float = 0.05
    low_quality_entry_penalty: float = 0.05
    missed_setup_regret: float = 0.05


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
            self._entry_stop_distance = 0.0
            self._stop_loss = 0.0
            self._take_profit = 0.0
            self._episode_step = 0
            self._last_exit_price = 0.0
            self._closed_r: list[float] = []
            self._regret_charged = False

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self.current_step = 0
            self._position = 0
            self._equity = self.config.initial_balance
            self._peak_equity = self.config.initial_balance
            self._entry_price = 0.0
            self._entry_stop_distance = 0.0
            self._stop_loss = 0.0
            self._take_profit = 0.0
            self._episode_step = 0
            self._last_exit_price = 0.0
            self._closed_r = []
            self._regret_charged = False
            self.ledger.records.clear()
            return self._observation(), {}

        def step(self, action: int):
            if action not in (0, 1, 2, 3):
                raise ValueError("action must be 0 (hold), 1 (long), 2 (short), or 3 (flatten)")
            if self.current_step >= len(self.bars):
                raise RuntimeError("step() called past the end of the episode; call reset()")
            episode_start = self.current_step == 0
            decision_bar = self.bars.iloc[self.current_step]
            previous = float(decision_bar["close"])
            old_position = self._position
            self._closed_r = []  # reset per-step closure ledger
            target = self._position
            if action == 1:
                target = 1
            elif action == 2:
                target = -1
            elif action == 3:
                target = 0
            turnover = abs(target - old_position)

            previous_equity = self._equity
            cost_fraction = self.config.transaction_cost
            extra_closes = 0

            # --- 1) Action-driven leg closure (flatten / reverse) at the
            # decision price ------------------------------------------------
            action_realized = 0.0
            if old_position != 0 and target != old_position:
                action_realized = self._realize_return(old_position, previous)

            # --- 2) Open a new leg (entry or the reversal leg) -------------
            if target != 0 and (old_position == 0 or target != old_position):
                self._open_leg(target, previous, decision_bar)
                if old_position == 0:
                    self._record_entry(target, previous, decision_bar)

            self.current_step += 1
            current_bar = self.bars.iloc[min(self.current_step, len(self.bars) - 1)]
            current_close = float(current_bar["close"])

            # --- 3) Managed exit (SL/TP/trailing) on the held leg ----------
            managed_realized = 0.0
            exit_price_for_equity: float | None = None
            if (
                self.config.use_managed_exits
                and self._position != 0
                and self._position == target
                and self.current_step < len(self.bars)
            ):
                managed_realized, exit_price = self._check_managed_exit(current_bar)
                if managed_realized != 0.0 or exit_price is not None:
                    extra_closes += 1
                    exit_price_for_equity = exit_price

            # --- 4) Episode end: force-flatten an open position ------------
            self._episode_step += 1
            terminated = self.current_step >= len(self.bars) - 1 or self._equity <= 0
            truncated = (
                self.config.episode_length_bars > 0
                and self._episode_step >= self.config.episode_length_bars
            )
            end_realized = 0.0
            if (terminated or truncated) and self._position != 0:
                end_realized = self._realize_return(self._position, current_close)
                self._position = 0
                extra_closes += 1
                exit_price_for_equity = current_close

            # --- 5) Equity update (realized PnL + floating mark) -----------
            realized_frac = action_realized + managed_realized + end_realized
            equity_change = previous_equity * realized_frac
            equity_change -= previous_equity * (turnover + extra_closes) * cost_fraction
            if self._position != 0 and exit_price_for_equity is None:
                # Still holding: mark-to-market over this bar (not rewarded in
                # the sparse modes; it only keeps equity/termination honest).
                equity_change += previous_equity * self._position * (current_close - previous) / max(previous, 1e-12)
            elif self._position != 0 and exit_price_for_equity is not None:
                # Reversal case where the new leg stays open: mark from entry.
                equity_change += previous_equity * self._position * (current_close - previous) / max(previous, 1e-12)
            self._equity = previous_equity + equity_change
            self._peak_equity = max(self._peak_equity, self._equity)

            # --- 6) Reward -------------------------------------------------
            if self.config.reward_type == "risk_adjusted":
                from slytrade.rl.rewards import RewardConfig, shaped_reward

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
                reward = realized_frac - (turnover + extra_closes) * cost_fraction
            elif self.config.reward_type == "r_multiple":
                reward = self._r_multiple_reward(decision_bar, opened=target != 0 and (old_position == 0 or target != old_position))
            else:
                reward = equity_change / max(previous_equity, 1e-12)

            return self._observation(), float(reward), terminated, truncated, {
                "equity": float(self._equity),
                "n_trades": len(self.ledger.records),
                "episode_start": episode_start,
            }

        # --- trade-management helpers ---------------------------------------
        def _open_leg(self, direction: int, price: float, bar: pd.Series) -> None:
            self._position = direction
            self._entry_price = price
            atr = float(bar.get("atr", 0.0) or 0.0)
            if atr <= 0:
                high = float(bar.get("high", price))
                low = float(bar.get("low", price))
                atr = max(high - low, price * 0.0005)
            stop_dist = max(atr * self.config.stop_loss_atr, price * 0.0002)
            target_dist = atr * self.config.take_profit_atr
            self._entry_stop_distance = stop_dist
            if direction > 0:
                self._stop_loss = price - stop_dist
                self._take_profit = price + target_dist
            else:
                self._stop_loss = price + stop_dist
                self._take_profit = price - target_dist

        def _realize_return(self, direction: int, exit_price: float) -> float:
            from slytrade.rl.rewards import r_from_fraction

            if self._entry_price > 0:
                realized = direction * (exit_price - self._entry_price) / self._entry_price
            else:
                realized = 0.0
            # Record the R-multiple of this closure (used by the production
            # reward) before the entry state is cleared.
            self._closed_r.append(r_from_fraction(realized, self._entry_price, self._entry_stop_distance))
            self._entry_price = 0.0
            self._entry_stop_distance = 0.0
            self._stop_loss = 0.0
            self._take_profit = 0.0
            return realized

        def _r_multiple_reward(self, decision_bar: pd.Series, *, opened: bool) -> float:
            """Production reward: realised R + ICT-footprint shaping.

            * realised R on every closure (sparse, risk-normalised),
            * entry-quality shaping: +bonus for high-confluence entries,
              −penalty for low-confluence entries,
            * missed-setup regret: a small −R when a high-confluence setup
              prints and the agent stays flat (kills the "never trade" trap).
            """
            from slytrade.rl.rewards import opening_cost_r

            reward = float(sum(self._closed_r))

            if opened:
                score = self._setup_score(decision_bar)
                if score >= self.config.setup_score_threshold:
                    reward += self.config.entry_quality_bonus
                else:
                    reward -= self.config.low_quality_entry_penalty
                reward -= opening_cost_r(
                    self.config.transaction_cost,
                    self._entry_price if self._entry_price > 0 else float(decision_bar["close"]),
                    self._entry_stop_distance if self._entry_stop_distance > 0 else float(decision_bar["close"]) * 0.0002,
                )
                self._regret_charged = True  # entered: don't charge regret this bar
                return reward

            # Flat: opportunity-cost regret for ignoring a high-confluence setup.
            if self._position == 0:
                score = self._setup_score(decision_bar)
                if score >= self.config.setup_score_threshold:
                    if not self._regret_charged:
                        reward -= self.config.missed_setup_regret
                        self._regret_charged = True
                else:
                    self._regret_charged = False
            return reward

        def _setup_score(self, bar: pd.Series) -> int:
            """ICT confluence score (mirrors the persona strategy's scorer).

            The footprint a professional trader waits for: market-structure
            breaks (BOS/CHOCH), liquidity sweeps, FVGs, order blocks, and
            premium/discount location. Returns the max of the long/short score.
            """
            long_score = 0
            short_score = 0
            premium_discount = float(bar.get("premium_discount", 0.0))
            trend = float(bar.get("trend_strength", 0.0))

            if float(bar.get("bos_dir", 0.0)) > 0:
                long_score += 2
            if float(bar.get("bos_dir", 0.0)) < 0:
                short_score += 2
            if float(bar.get("choch_dir", 0.0)) > 0:
                long_score += 1
            if float(bar.get("choch_dir", 0.0)) < 0:
                short_score += 1
            if float(bar.get("liquidity_sweep", 0.0)) < 0:
                long_score += 1
            if float(bar.get("liquidity_sweep", 0.0)) > 0:
                short_score += 1
            if float(bar.get("fvg_bullish", 0.0)) > 0:
                long_score += 1
            if float(bar.get("fvg_bearish", 0.0)) > 0:
                short_score += 1
            if float(bar.get("order_block_bullish", 0.0)) > 0:
                long_score += 1
            if float(bar.get("order_block_bearish", 0.0)) > 0:
                short_score += 1
            if premium_discount <= -0.15:
                long_score += 1
            if premium_discount >= 0.15:
                short_score += 1
            if trend > 0:
                long_score += 1
            if trend < 0:
                short_score += 1
            return max(long_score, short_score)

        def _check_managed_exit(self, bar: pd.Series) -> tuple[float, float | None]:
            """Return (realized_return, exit_price) if SL/TP was hit this bar."""
            high = float(bar.get("tick_mid_high", bar.get("high", 0.0)) or 0.0)
            low = float(bar.get("tick_mid_low", bar.get("low", 0.0)) or 0.0)
            if high <= 0 or low <= 0:
                return 0.0, None
            direction = self._position
            if direction > 0:
                sl_hit = low <= self._stop_loss
                tp_hit = high >= self._take_profit
            else:
                sl_hit = high >= self._stop_loss
                tp_hit = low <= self._take_profit

            if not sl_hit and not tp_hit:
                return 0.0, None

            # Conservative same-bar rule (mirrors the backtest engine): if both
            # are touched, the stop-loss wins.
            if sl_hit:
                exit_price = self._stop_loss
            else:
                exit_price = self._take_profit
            realized = self._realize_return(direction, exit_price)
            self._position = 0
            self._last_exit_price = exit_price
            return realized, exit_price

        def _record_entry(self, direction: int, price: float, bar: pd.Series) -> None:
            intent = OrderIntent(
                symbol=str(bar["symbol"]),
                side=Side.BUY if direction > 0 else Side.SELL,
                volume=min(self.config.max_position_volume, self.config.risk_per_trade),
                reason="rl_entry",
            )
            self.ledger.record_fill(
                intent,
                volume=intent.volume,
                price=price,
                commission=self.config.transaction_cost * intent.volume,
                realized_pnl=0.0,
                event_time=pd.Timestamp(bar["time"]).to_pydatetime(),
            )

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
