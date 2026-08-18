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
    # Time-based exit in bars (0 = off). The validated per-timeframe profiles
    # use this (e.g. M15 holds 60 bars), and the RL must play the SAME game as
    # the champion it is measured against — otherwise it is optimising a
    # different, worse exit structure.
    max_bars_in_trade: int = 0
    # Production reward (reward_type="r_multiple") — the unit is R, the risk
    # at entry. Entry/regret shaping is driven by the ICT footprint score.
    setup_score_threshold: int = 4
    entry_quality_bonus: float = 0.05
    low_quality_entry_penalty: float = 0.05
    missed_setup_regret: float = 0.05
    # Shaping master switch. The bonus/penalty/regret terms reward ACTIVITY,
    # and measured on real data they drove the agent to overtrade (~150 trades
    # per episode) and lose ~0.5R/trade with no directional edge — the exact
    # "in-sample +30%, out-of-sample -62%" signature. OFF by default so
    # r_multiple is pure realised R minus the true opening cost.
    shaping_enabled: bool = False
    # Activity brake: hard cap on entries per episode (0 = unlimited). Past the
    # cap the agent may only hold or flatten — it can never open new risk. A
    # sane value forces the agent to pick its best few setups, like a pro.
    max_trades_per_episode: int = 0
    # Candidate-bar masking (the "RL as a filter" methodology). When enabled,
    # the env only accepts new entries on bars where the persona would consider
    # a setup (the candidate mask); everywhere else the agent may only hold or
    # flatten. This is NOT a hardcoded trade cap — the candidates ARE the
    # dynamic opportunities, so the agent's selectivity is bounded by the
    # market's own setups, exactly like a pro who only looks for their pattern.
    mask_to_candidates: bool = False
    # Round-trip trading cost in R, charged once per opened leg (spread +
    # commission + slippage, measured per timeframe from real data: M15 ~0.043,
    # H1 ~0.021, H4 ~0.011, M5 ~0.076, M1 ~0.176). The old price-fraction cost
    # (0.0002) converted to ~0.16R on gold — ~4x the real cost — which is why
    # even the champion barely broke even inside the env. This must match the
    # backtest's net-of-cost economics or the RL is learning a different game.
    round_trip_cost_r: float = 0.04
    # Entry cooldown in bars, mirroring the persona's cooldown gate. The env
    # exposes "cooldown remaining" as an observation scalar so a learned policy
    # can reproduce the stateful entry gating (the part of the champion's edge a
    # stateless observation cannot carry).
    entry_cooldown_bars: int = 10


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
            candidate_mask: np.ndarray | None = None,
            **_: object,
        ):
            if features.empty or len(features) != len(bars):
                raise ValueError("features and bars must be non-empty and aligned")
            required = {"time", "symbol", "open", "high", "low", "close"}
            if not required.issubset(bars.columns):
                raise ValueError(f"bars missing required columns: {sorted(required.difference(bars.columns))}")
            super().__init__()
            # The env indexes rows positionally (`.iloc`), so a RangeIndex is
            # all it needs — reset_index(drop=True) would copy a multi-GB frame.
            self.features = features if isinstance(features.index, pd.RangeIndex) else features.reset_index(drop=True)
            self.bars = bars if isinstance(bars.index, pd.RangeIndex) else bars.reset_index(drop=True)
            self.config = config or RLEnvironmentConfig()
            self.mode_vector = mode_vector
            self.candidate_mask = (
                np.asarray(candidate_mask, dtype=np.float32) if candidate_mask is not None else None
            )
            # Precompute the full observation matrix once (float32) so each
            # step's observation is a single numpy row slice instead of a
            # pandas Series -> numpy conversion. When the frame is already
            # float32 this is a zero-copy view of its buffer.
            self._feature_matrix = self.features.to_numpy(dtype=np.float32)
            self._mode_vector32 = np.asarray(mode_vector, dtype=np.float32) if mode_vector is not None else None
            # Agent state appended to the observation (the part of the Markov
            # state the raw features do not carry): position, bars-in-trade,
            # entries-used, last-exit-R, episode progress. Without these the
            # policy cannot learn "I am already in a trade / I just lost" —
            # which is exactly the statefulness the persona exploits.
            self._state_vector = np.zeros(6, dtype=np.float32)
            self._n_state = 6
            shape = (
                len(self.features.columns)
                + (len(mode_vector) if mode_vector is not None else 0)
                + self._n_state
            )
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
            self._entries_this_episode = 0
            self._bars_in_trade = 0

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
            self._entries_this_episode = 0
            self._bars_in_trade = 0
            self._bars_since_entry = 10_000
            self.ledger.records.clear()
            self._update_state_vector()
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
            # Activity brake: past the entry cap the agent may only hold or
            # flatten — it can never open new risk. This forces the policy to
            # spend its scarce entries on its best setups instead of churning.
            if (
                target != 0
                and (old_position == 0 or target != old_position)
                and self.config.max_trades_per_episode > 0
                and self._entries_this_episode >= self.config.max_trades_per_episode
            ):
                target = old_position
            # Candidate-bar masking: outside the persona's candidate setups the
            # agent may only hold or flatten. The candidates ARE the dynamic
            # opportunities, so this bounds selectivity by the market's own
            # setups rather than a hardcoded number.
            if (
                self.config.mask_to_candidates
                and self.candidate_mask is not None
                and target != 0
                and (old_position == 0 or target != old_position)
                and float(self.candidate_mask[self.current_step]) == 0.0
            ):
                target = old_position
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
                    self._bars_since_entry = 0

            self.current_step += 1
            self._bars_since_entry = min(self._bars_since_entry + 1, 10_000)
            current_bar = self.bars.iloc[min(self.current_step, len(self.bars) - 1)]
            current_close = float(current_bar["close"])

            # --- 3) Managed exit (SL/TP/trailing/time) on the held leg -----
            managed_realized = 0.0
            exit_price_for_equity: float | None = None
            if (
                self.config.use_managed_exits
                and self._position != 0
                and self._position == target
                and self.current_step < len(self.bars)
            ):
                self._bars_in_trade += 1
                managed_realized, exit_price = self._check_managed_exit(current_bar)
                if managed_realized != 0.0 or exit_price is not None:
                    extra_closes += 1
                    exit_price_for_equity = exit_price
                elif (
                    self.config.max_bars_in_trade > 0
                    and self._bars_in_trade >= self.config.max_bars_in_trade
                ):
                    # Time-based exit at the current close — same structure the
                    # validated per-timeframe profiles use.
                    managed_realized = self._realize_return(self._position, current_close)
                    self._position = 0
                    extra_closes += 1
                    exit_price_for_equity = current_close

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

            self._update_state_vector()
            return self._observation(), float(reward), terminated, truncated, {
                "equity": float(self._equity),
                "n_trades": len(self.ledger.records),
                "episode_start": episode_start,
            }

        # --- trade-management helpers ---------------------------------------
        def _open_leg(self, direction: int, price: float, bar: pd.Series) -> None:
            self._position = direction
            self._entry_price = price
            self._bars_in_trade = 0
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
            """Production reward: realised R minus the true round-trip cost.

            With ``shaping_enabled`` it additionally adds entry-quality shaping
            (+bonus for high-confluence entries, −penalty for low-quality ones,
            −regret for staying flat on a setup). That shaping rewards activity,
            so it is OFF by default: the default reward is purely the realised
            R of closed legs minus the measured round-trip cost in R — the only
            terms that correspond to money in the account.
            """
            reward = float(sum(self._closed_r))

            if opened:
                if self.config.shaping_enabled:
                    score = self._setup_score(decision_bar)
                    if score >= self.config.setup_score_threshold:
                        reward += self.config.entry_quality_bonus
                    else:
                        reward -= self.config.low_quality_entry_penalty
                # Real cost: spread + commission + slippage as a round trip in R
                # (per timeframe), instead of a price-fraction that blew up to
                # ~4x the true cost on gold.
                reward -= float(self.config.round_trip_cost_r)
                self._regret_charged = True  # entered: don't charge regret this bar
                return reward

            # Flat: opportunity-cost regret for ignoring a high-confluence setup.
            if self._position == 0 and self.config.shaping_enabled:
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
            self._entries_this_episode += 1
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
            index = min(self.current_step, self._feature_matrix.shape[0] - 1)
            row = self._feature_matrix[index]
            parts = [row]
            if self._mode_vector32 is not None:
                parts.append(self._mode_vector32)
            parts.append(self._state_vector)
            return np.concatenate(parts)

        def _update_state_vector(self) -> None:
            """Refresh the agent-state part of the observation."""
            n = len(self.bars)
            max_bars = self.config.max_bars_in_trade or 200
            max_entries = self.config.max_trades_per_episode or 200
            last_r = self._closed_r[-1] if self._closed_r else 0.0
            cooldown = max(1, self.config.entry_cooldown_bars)
            self._state_vector[0] = float(np.clip(self._position, -1.0, 1.0))
            self._state_vector[1] = min(float(self._bars_in_trade) / max_bars, 1.0)
            self._state_vector[2] = min(float(self._entries_this_episode) / max_entries, 1.0)
            self._state_vector[3] = float(np.clip(last_r, -5.0, 5.0)) / 5.0
            self._state_vector[4] = float(self.current_step) / max(n, 1)
            # Cooldown remaining: 0 = eligible to enter, rising to 1 = fully in
            # cooldown. Mirrors the persona's stateful entry gating.
            remaining = max(0.0, cooldown - float(self._bars_since_entry)) / cooldown
            self._state_vector[5] = float(remaining)

else:

    class SlyTradeRLEnvironment:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError("SlyTradeRLEnvironment requires the optional 'rl' dependencies")
