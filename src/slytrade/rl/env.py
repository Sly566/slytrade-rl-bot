"""SlyTrade Gymnasium Environment — wraps backtest engine as standard RL env.

The agent observes market state and takes actions (enter/exit/hold/sizing).
The environment walks through historical M1 bars using the same signal
pipeline and backtest engine as the live trader.

Observation Space (Box):
  - Market features: OHLC normalized, ATR, structure flags, zone proximity
  - Position state: direction, entry offset, P&L R-multiple, bars held
  - Account state: equity curve, current drawdown, win rate

Action Space (MultiDiscrete):
  - action_type: 0=HOLD, 1=CLOSE, 2=ENTER_LONG, 3=ENTER_SHORT
  - size_level: 0=min_lot(0.01), 1=working_lot(0.04), 2=2x_working(0.08)
  - sl_mult: 0=1.0x_ATR, 1=1.5x_ATR, 2=2.0x_ATR, 3=2.5x_ATR
  - tp_mult: 0=0.5R, 1=1.0R, 2=1.5R, 3=2.0R, 4=2.5R
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..backtest.specs import AccountSpec, SymbolSpec, spec_for_symbol
from ..data.features import DEFAULT_CONFIG, process_bars
from ..data.mtf_align import _asof_merge, _prep_htf_frame
from ..data.time import timeframe_timedelta
from ..strategy.config import StrategyConfig, rl_training_persona
from ..strategy.signals import _evaluate_row


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Observation vector layout
# Market features (normalized)
_OBS_BID = 0          # current bid price (normalized)
_OBS_ASK = 1          # current ask price (normalized)
_OBS_ATR = 2          # ATR(14) normalized by price
_OBS_SPREAD = 3       # spread normalized by ATR
# Structure flags (0/1)
_OBS_BULL_DISP = 4    # M1 bull displacement
_OBS_BEAR_DISP = 5    # M1 bear displacement
_OBS_BOS_UP = 6       # M1 minor BOS up
_OBS_BOS_DN = 7       # M1 minor BOS down
_OBS_CHOCH_UP = 8     # M1 minor CHoCH up
_OBS_CHOCH_DN = 9     # M1 minor CHoCH down
_OBS_M5_BULL_DISP = 10
_OBS_M5_BEAR_DISP = 11
_OBS_M5_BOS_UP = 12
_OBS_M5_BOS_DN = 13
_OBS_M5_CHOCH_UP = 14
_OBS_M5_CHOCH_DN = 15
# M15 structure (broader intraday blanket)
_OBS_M15_BULL_DISP = 16
_OBS_M15_BEAR_DISP = 17
_OBS_M15_BOS_UP = 18
_OBS_M15_BOS_DN = 19
_OBS_M15_CHOCH_UP = 20
_OBS_M15_CHOCH_DN = 21
_OBS_M15_MAJOR_CHOCH_UP = 22
_OBS_M15_MAJOR_CHOCH_DN = 23
# Zone proximity (distance to nearest OB/FVG normalized by ATR)
_OBS_OB_PROX = 24     # distance to nearest unmitigated OB
_OBS_FVG_PROX = 25    # distance to nearest unmitigated FVG
_OBS_SWEEP_PROX = 26  # distance to nearest liquidity sweep level
# Position state
_OBS_POS_DIR = 27     # +1 long, -1 short, 0 flat
_OBS_POS_R = 28       # current P&L in R-multiples
_OBS_POS_BARS = 29    # bars held (normalized by time_stop)
_OBS_POS_AGE = 30     # time since entry (normalized)
# Account state
_OBS_EQUITY_CURVE = 31  # equity / starting_equity
_OBS_DRAWDOWN = 32      # current drawdown from peak
_OBS_WIN_RATE = 33      # recent win rate (last 20 trades)
# Killzone
_OBS_KZ_ASIA = 34
_OBS_KZ_LONDON = 35
_OBS_KZ_NY = 36
# Time features
_OBS_HOUR_SIN = 37    # hour encoded as sin
_OBS_HOUR_COS = 38    # hour encoded as cos

OBS_DIM = 39

# Action space
ACT_HOLD = 0
ACT_CLOSE = 1
ACT_ENTER_LONG = 2
ACT_ENTER_SHORT = 3

# Size levels
SIZE_MIN = 0       # 0.01 lots
SIZE_WORKING = 1   # 0.04 lots
SIZE_DOUBLE = 2    # 0.08 lots

# SL/TP multipliers
SL_MULTS = [1.0, 1.5, 2.0, 2.5]   # × ATR
TP_MULTS = [0.5, 1.0, 1.5, 2.0, 2.5]  # × R


class SlyTradeEnv(gym.Env):
    """Gymnasium environment wrapping the SlyTrade backtest engine.

    The agent walks through historical M1 bars, observing market state
    and taking actions (enter/exit/hold). The environment computes P&L
    using the same signal pipeline as the live trader.

    Args:
        aligned_df: Pre-aligned M1 bars with HTF features (from backtest data)
        cfg: Strategy configuration (default: rl_training_persona)
        spec: Symbol specification
        acct: Account specification
        max_bars: Maximum bars per episode (default: all bars)
        time_stop_bars: Time-stop in bars (default: 60)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        aligned_df: pd.DataFrame,
        cfg: StrategyConfig | None = None,
        spec: SymbolSpec | None = None,
        acct: AccountSpec | None = None,
        *,
        max_bars: int | None = None,
        time_stop_bars: int = 60,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Data — convert to numpy for fast O(1) access (100x faster than pandas iloc)
        aligned_df = aligned_df.reset_index(drop=True)
        self.n_bars = min(max_bars or len(aligned_df), len(aligned_df))

        # Pre-extract columns as numpy arrays
        _COL_MAP = {
            "close": "close", "atr_14": "atr_14", "time": "time",
            "bull_disp": "bull_disp", "bear_disp": "bear_disp",
            "minor_bos_up": "minor_bos_up", "minor_bos_dn": "minor_bos_dn",
            "minor_choch_up": "minor_choch_up", "minor_choch_dn": "minor_choch_dn",
            "M5_bull_disp": "M5_bull_disp", "M5_bear_disp": "M5_bear_disp",
            "M5_minor_bos_up": "M5_minor_bos_up", "M5_minor_bos_dn": "M5_minor_bos_dn",
            "M5_minor_choch_up": "M5_minor_choch_up", "M5_minor_choch_dn": "M5_minor_choch_dn",
            "M15_bull_disp": "M15_bull_disp", "M15_bear_disp": "M15_bear_disp",
            "M15_minor_bos_up": "M15_minor_bos_up", "M15_minor_bos_dn": "M15_minor_bos_dn",
            "M15_minor_choch_up": "M15_minor_choch_up", "M15_minor_choch_dn": "M15_minor_choch_dn",
            "M15_major_choch_up": "M15_major_choch_up", "M15_major_choch_dn": "M15_major_choch_dn",
        }
        self._col_arrays = {}
        for key, col in _COL_MAP.items():
            if col in aligned_df.columns:
                if col == "time":
                    self._col_arrays[key] = aligned_df[col].values  # keep as datetime64
                elif aligned_df[col].dtype == bool:
                    self._col_arrays[key] = aligned_df[col].values.astype(np.float32)
                else:
                    self._col_arrays[key] = pd.to_numeric(aligned_df[col], errors="coerce").fillna(0.0).values.astype(np.float64)
            else:
                self._col_arrays[key] = np.zeros(len(aligned_df), dtype=np.float64)

        # Keep aligned for _evaluate_row compatibility (lightweight reference)
        self.aligned = aligned_df

        # Config
        self.cfg = cfg or rl_training_persona()
        self.spec = spec or spec_for_symbol("XAUUSDm")
        self.acct = acct or AccountSpec(
            starting_equity=2000.0, currency="ZAR",
            leverage=2000, fx_to_account={"USD": 18.5},
        )
        self.time_stop_bars = time_stop_bars

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        # MultiDiscrete: [action_type, size_level, sl_mult, tp_mult]
        self.action_space = spaces.MultiDiscrete([4, 3, 4, 5])

        # State
        self._reset_state()

    def _reset_state(self):
        """Reset internal state for a new episode."""
        self._bar_idx = 0
        self._state: dict = {}  # signal engine state
        self._equity = self.acct.starting_equity
        self._peak_equity = self._equity
        self._starting_equity = self._equity

        # Position state
        self._pos_dir = 0        # 0=flat, +1=long, -1=short
        self._pos_entry = 0.0
        self._pos_sl = 0.0
        self._pos_tp = 0.0
        self._pos_lots = 0.0
        self._pos_bars = 0
        self._pos_risk_per_unit = 0.0

        # Trade history
        self._trades: list[dict] = []
        self._recent_wins: list[bool] = []  # last 20 trades

    def _get_obs(self) -> np.ndarray:
        """Build observation vector from current bar and position state.
        Uses pre-extracted numpy arrays for O(1) access (~100x faster than pandas iloc).
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        i = self._bar_idx
        if i >= self.n_bars:
            return obs

        ca = self._col_arrays
        price = ca["close"][i]
        atr = ca["atr_14"][i]
        if not np.isfinite(atr) or atr <= 0:
            atr = 0.0

        price_norm = price / 1000.0 if price > 0 else 0.0
        obs[_OBS_BID] = price_norm
        obs[_OBS_ASK] = price_norm + 0.0002
        obs[_OBS_ATR] = atr / max(price, 1.0)
        obs[_OBS_SPREAD] = 0.2 / max(atr, 0.001)

        # Structure flags — direct numpy array indexing
        obs[_OBS_BULL_DISP] = ca["bull_disp"][i]
        obs[_OBS_BEAR_DISP] = ca["bear_disp"][i]
        obs[_OBS_BOS_UP] = ca["minor_bos_up"][i]
        obs[_OBS_BOS_DN] = ca["minor_bos_dn"][i]
        obs[_OBS_CHOCH_UP] = ca["minor_choch_up"][i]
        obs[_OBS_CHOCH_DN] = ca["minor_choch_dn"][i]
        obs[_OBS_M5_BULL_DISP] = ca["M5_bull_disp"][i]
        obs[_OBS_M5_BEAR_DISP] = ca["M5_bear_disp"][i]
        obs[_OBS_M5_BOS_UP] = ca["M5_minor_bos_up"][i]
        obs[_OBS_M5_BOS_DN] = ca["M5_minor_bos_dn"][i]
        obs[_OBS_M5_CHOCH_UP] = ca["M5_minor_choch_up"][i]
        obs[_OBS_M5_CHOCH_DN] = ca["M5_minor_choch_dn"][i]
        obs[_OBS_M15_BULL_DISP] = ca["M15_bull_disp"][i]
        obs[_OBS_M15_BEAR_DISP] = ca["M15_bear_disp"][i]
        obs[_OBS_M15_BOS_UP] = ca["M15_minor_bos_up"][i]
        obs[_OBS_M15_BOS_DN] = ca["M15_minor_bos_dn"][i]
        obs[_OBS_M15_CHOCH_UP] = ca["M15_minor_choch_up"][i]
        obs[_OBS_M15_CHOCH_DN] = ca["M15_minor_choch_dn"][i]
        obs[_OBS_M15_MAJOR_CHOCH_UP] = ca["M15_major_choch_up"][i]
        obs[_OBS_M15_MAJOR_CHOCH_DN] = ca["M15_major_choch_dn"][i]

        # Zone proximity placeholders
        obs[_OBS_OB_PROX] = 0.5
        obs[_OBS_FVG_PROX] = 0.5
        obs[_OBS_SWEEP_PROX] = 0.5

        # Position state
        obs[_OBS_POS_DIR] = float(self._pos_dir)
        if self._pos_dir != 0 and self._pos_risk_per_unit > 0:
            r_dist = (price - self._pos_entry) if self._pos_dir == 1 else (self._pos_entry - price)
            obs[_OBS_POS_R] = r_dist / self._pos_risk_per_unit
        obs[_OBS_POS_BARS] = self._pos_bars / max(self.time_stop_bars, 1)
        obs[_OBS_POS_AGE] = self._pos_bars / max(self.time_stop_bars, 1)

        # Account state
        obs[_OBS_EQUITY_CURVE] = self._equity / max(self._starting_equity, 1.0)
        obs[_OBS_DRAWDOWN] = (self._peak_equity - self._equity) / max(self._peak_equity, 1.0)
        if self._recent_wins:
            obs[_OBS_WIN_RATE] = sum(self._recent_wins[-20:]) / len(self._recent_wins[-20:])

        # Killzone — extract hour from numpy datetime64
        try:
            ts_val = ca["time"][i]
            hour = pd.Timestamp(ts_val).hour
            obs[_OBS_KZ_ASIA] = 1.0 if 0 <= hour < 8 else 0.0
            obs[_OBS_KZ_LONDON] = 1.0 if 7 <= hour < 16 else 0.0
            obs[_OBS_KZ_NY] = 1.0 if 12 <= hour < 21 else 0.0
            obs[_OBS_HOUR_SIN] = np.sin(2 * np.pi * hour / 24.0)
            obs[_OBS_HOUR_COS] = np.cos(2 * np.pi * hour / 24.0)
        except Exception:
            pass

        return obs

    def _get_info(self) -> dict:
        """Return info dict for debugging/logging."""
        return {
            "bar_idx": self._bar_idx,
            "equity": self._equity,
            "drawdown": (self._peak_equity - self._equity) / max(self._peak_equity, 1.0),
            "pos_dir": self._pos_dir,
            "pos_bars": self._pos_bars,
            "n_trades": len(self._trades),
            "win_rate": sum(self._recent_wins[-20:]) / max(len(self._recent_wins[-20:]), 1),
        }

    def reset(self, *, seed=None, options=None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one step: process action, advance one bar, compute reward.

        Args:
            action: MultiDiscrete array [action_type, size_level, sl_mult, tp_mult]

        Returns:
            obs, reward, terminated, truncated, info
        """
        action_type = int(action[0])
        size_level = int(action[1])
        sl_mult_idx = int(action[2])
        tp_mult_idx = int(action[3])

        i = self._bar_idx
        ca = self._col_arrays
        price = ca["close"][i]
        atr = ca["atr_14"][i]
        if not np.isfinite(atr) or atr <= 0:
            atr = 0.0
        bid = price
        ask = price + 0.0002

        reward = 0.0
        trade_closed = False
        close_pnl = 0.0

        # --- Process action ---
        if action_type == ACT_CLOSE and self._pos_dir != 0:
            # Close position
            if self._pos_dir == 1:
                close_pnl = (bid - self._pos_entry) * self._pos_lots * self.spec.contract_size
            else:
                close_pnl = (self._pos_entry - ask) * self._pos_lots * self.spec.contract_size
            # Convert to account currency
            close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
            self._equity += close_pnl_acct
            self._peak_equity = max(self._peak_equity, self._equity)
            self._recent_wins.append(close_pnl_acct > 0)
            self._trades.append({
                "entry": self._pos_entry,
                "exit": price,
                "dir": self._pos_dir,
                "lots": self._pos_lots,
                "pnl": close_pnl_acct,
                "bars": self._pos_bars,
            })
            self._pos_dir = 0
            self._pos_entry = 0.0
            self._pos_sl = 0.0
            self._pos_tp = 0.0
            self._pos_lots = 0.0
            self._pos_bars = 0
            self._pos_risk_per_unit = 0.0
            trade_closed = True

        elif action_type == ACT_ENTER_LONG and self._pos_dir == 0:
            # Enter long
            lots = [0.01, 0.04, 0.08][size_level]
            sl_dist = SL_MULTS[sl_mult_idx] * max(atr, 0.01)
            tp_dist_mult = TP_MULTS[tp_mult_idx]
            self._pos_dir = 1
            self._pos_entry = ask
            self._pos_sl = ask - sl_dist
            self._pos_tp = ask + tp_dist_mult * sl_dist  # TP = tp_mult × R
            self._pos_lots = lots
            self._pos_bars = 0
            self._pos_risk_per_unit = sl_dist

        elif action_type == ACT_ENTER_SHORT and self._pos_dir == 0:
            # Enter short
            lots = [0.01, 0.04, 0.08][size_level]
            sl_dist = SL_MULTS[sl_mult_idx] * max(atr, 0.01)
            tp_dist_mult = TP_MULTS[tp_mult_idx]
            self._pos_dir = -1
            self._pos_entry = bid
            self._pos_sl = bid + sl_dist
            self._pos_tp = bid - tp_dist_mult * sl_dist
            self._pos_lots = lots
            self._pos_bars = 0
            self._pos_risk_per_unit = sl_dist

        # --- Check SL/TP/time-stop on existing position ---
        if self._pos_dir != 0:
            self._pos_bars += 1

            # Check SL hit
            if self._pos_dir == 1 and bid <= self._pos_sl:
                close_pnl = (self._pos_sl - self._pos_entry) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(False)
                self._trades.append({"entry": self._pos_entry, "exit": self._pos_sl, "dir": 1,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "SL"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

            elif self._pos_dir == -1 and ask >= self._pos_sl:
                close_pnl = (self._pos_entry - self._pos_sl) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(False)
                self._trades.append({"entry": self._pos_entry, "exit": self._pos_sl, "dir": -1,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "SL"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

            # Check TP hit
            elif self._pos_dir == 1 and bid >= self._pos_tp:
                close_pnl = (self._pos_tp - self._pos_entry) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(True)
                self._trades.append({"entry": self._pos_entry, "exit": self._pos_tp, "dir": 1,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "TP"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

            elif self._pos_dir == -1 and ask <= self._pos_tp:
                close_pnl = (self._pos_entry - self._pos_tp) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(True)
                self._trades.append({"entry": self._pos_entry, "exit": self._pos_tp, "dir": -1,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "TP"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

            # Time-stop
            elif self._pos_bars >= self.time_stop_bars:
                if self._pos_dir == 1:
                    close_pnl = (bid - self._pos_entry) * self._pos_lots * self.spec.contract_size
                else:
                    close_pnl = (self._pos_entry - ask) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(close_pnl_acct > 0)
                self._trades.append({"entry": self._pos_entry, "exit": price, "dir": self._pos_dir,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "TIME_STOP"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

        # --- Compute reward ---
        # Reward = log return + drawdown penalty + trade bonus
        equity_frac = self._equity / max(self._starting_equity, 1.0)
        drawdown = (self._peak_equity - self._equity) / max(self._peak_equity, 1.0)

        # Log return component (per-step equity change)
        if len(self._trades) > 0 and trade_closed:
            last_pnl = self._trades[-1]["pnl"]
            reward = last_pnl / max(self._starting_equity, 1.0) * 100  # scale up
            # Bonus for wins, penalty for losses
            if last_pnl > 0:
                reward += 0.1
            else:
                reward -= 0.1
        else:
            # Small holding reward/penalty based on unrealized P&L
            if self._pos_dir != 0 and self._pos_risk_per_unit > 0:
                r_dist = (price - self._pos_entry) if self._pos_dir == 1 else (self._pos_entry - price)
                cur_r = r_dist / self._pos_risk_per_unit
                reward = cur_r * 0.01  # small per-step reward for being in profit

        # Drawdown penalty
        if drawdown > 0.05:  # >5% drawdown
            reward -= drawdown * 0.5

        # Advance to next bar
        self._bar_idx += 1
        terminated = self._bar_idx >= self.n_bars
        truncated = self._equity <= self._starting_equity * 0.5  # 50% equity = stop

        # Update signal engine state (for feature computation on next step)
        if not terminated:
            # Skip _evaluate_row — observation is built directly from numpy arrays
            pass

        obs = self._get_obs()
        info = self._get_info()
        info["trade_closed"] = trade_closed
        info["close_pnl"] = close_pnl if trade_closed else 0.0

        if self.render_mode == "human":
            self._render_human(info)

        return obs, reward, terminated, truncated, info

    def _render_human(self, info: dict):
        """Print current state to terminal."""
        if info.get("trade_closed"):
            pnl = info.get("close_pnl", 0.0)
            print(f"  bar={self._bar_idx} eq={self._equity:.0f} "
                  f"dd={info['drawdown']:.1%} trades={info['n_trades']} "
                  f"wr={info['win_rate']:.0%} pnl={pnl:+.2f}")

    def get_metrics(self) -> dict:
        """Compute backtest metrics from trade history."""
        if not self._trades:
            return {"n_trades": 0}

        pnls = [t["pnl"] for t in self._trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

        # Sharpe ratio (annualized, assuming 252 trading days × 24 hours)
        if len(pnls) > 1:
            returns = np.array(pnls) / self._starting_equity
            sharpe = np.mean(returns) / max(np.std(returns), 1e-9) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0

        # Max drawdown
        equity_curve = np.cumsum([self._starting_equity] + pnls)
        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        return {
            "n_trades": len(pnls),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "final_equity": self._equity,
            "return_pct": (self._equity - self._starting_equity) / self._starting_equity,
        }
