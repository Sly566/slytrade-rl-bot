"""Model serving — load trained RL model and use it in live trading.

The trained model acts as a signal filter: it observes market state and
decides whether to take or skip each signal from the rule-based engine.

Usage in live trader:
    from slytrade.rl.serve import RLFilter

    rl_filter = RLFilter("models/ppo_XAUUSDm_final.zip")

    # In _handle_signal:
    if rl_filter.should_skip(obs, signal):
        return  # skip this signal
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from stable_baselines3 import PPO, SAC, A2C
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from .env import (
    OBS_DIM,
    ACT_HOLD,
    ACT_CLOSE,
    ACT_ENTER_LONG,
    ACT_ENTER_SHORT,
)


class RLFilter:
    """Load a trained RL model and use it to filter signals.

    The model predicts: given current market state + proposed signal,
    should we take it (ENTER) or skip it (HOLD)?

    Args:
        model_path: Path to trained model (.zip)
        algo: Algorithm used (ppo, sac, a2c)
        threshold: Confidence threshold for taking signals (0.0-1.0)
    """

    def __init__(
        self,
        model_path: str,
        algo: str = "ppo",
        threshold: float = 0.5,
    ):
        if not HAS_SB3:
            raise ImportError(
                "stable-baselines3 not installed. "
                "Run: pip install 'slytrade-rl-bot[rl]'"
            )

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        algo_cls = {"ppo": PPO, "sac": SAC, "a2c": A2C}[algo.lower()]
        self.model = algo_cls.load(str(path))
        self.threshold = threshold
        self._obs = np.zeros(OBS_DIM, dtype=np.float32)

    def build_obs(
        self,
        row: Any,
        pos_dir: int = 0,
        pos_entry: float = 0.0,
        pos_risk: float = 0.0,
        pos_bars: int = 0,
        equity: float = 2000.0,
        peak_equity: float = 2000.0,
        starting_equity: float = 2000.0,
        recent_wins: list[bool] | None = None,
        time_stop_bars: int = 60,
    ) -> np.ndarray:
        """Build observation vector from current market state.

        This mirrors the observation construction in SlyTradeEnv._get_obs().
        """
        from .env import (
            _OBS_BID, _OBS_ASK, _OBS_ATR, _OBS_SPREAD,
            _OBS_BULL_DISP, _OBS_BEAR_DISP, _OBS_BOS_UP, _OBS_BOS_DN,
            _OBS_CHOCH_UP, _OBS_CHOCH_DN,
            _OBS_M5_BULL_DISP, _OBS_M5_BEAR_DISP,
            _OBS_M5_BOS_UP, _OBS_M5_BOS_DN,
            _OBS_M5_CHOCH_UP, _OBS_M5_CHOCH_DN,
            _OBS_OB_PROX, _OBS_FVG_PROX, _OBS_SWEEP_PROX,
            _OBS_POS_DIR, _OBS_POS_R, _OBS_POS_BARS, _OBS_POS_AGE,
            _OBS_EQUITY_CURVE, _OBS_DRAWDOWN, _OBS_WIN_RATE,
            _OBS_KZ_ASIA, _OBS_KZ_LONDON, _OBS_KZ_NY,
            _OBS_HOUR_SIN, _OBS_HOUR_COS,
        )

        obs = self._obs
        obs[:] = 0.0

        price = float(row.get("close", 0.0))
        atr = float(row.get("atr_14", 0.0)) if hasattr(row, "get") else 0.0
        if hasattr(row, "atr_14"):
            atr = float(row.atr_14) if not np.isnan(row.atr_14) else 0.0

        price_norm = price / 1000.0 if price > 0 else 0.0
        obs[_OBS_BID] = price_norm
        obs[_OBS_ASK] = price_norm + 0.0002
        obs[_OBS_ATR] = atr / max(price, 1.0)
        obs[_OBS_SPREAD] = 0.2 / max(atr, 0.001)

        # Structure flags
        for col, idx in [
            ("bull_disp", _OBS_BULL_DISP), ("bear_disp", _OBS_BEAR_DISP),
            ("minor_bos_up", _OBS_BOS_UP), ("minor_bos_dn", _OBS_BOS_DN),
            ("minor_choch_up", _OBS_CHOCH_UP), ("minor_choch_dn", _OBS_CHOCH_DN),
            ("M5_bull_disp", _OBS_M5_BULL_DISP), ("M5_bear_disp", _OBS_M5_BEAR_DISP),
            ("M5_minor_bos_up", _OBS_M5_BOS_UP), ("M5_minor_bos_dn", _OBS_M5_BOS_DN),
            ("M5_minor_choch_up", _OBS_M5_CHOCH_UP), ("M5_minor_choch_dn", _OBS_M5_CHOCH_DN),
        ]:
            try:
                obs[idx] = 1.0 if bool(row.get(col, False)) else 0.0
            except Exception:
                pass

        # Position state
        obs[_OBS_POS_DIR] = float(pos_dir)
        if pos_dir != 0 and pos_risk > 0:
            r_dist = (price - pos_entry) if pos_dir == 1 else (pos_entry - price)
            obs[_OBS_POS_R] = r_dist / pos_risk
        obs[_OBS_POS_BARS] = pos_bars / max(time_stop_bars, 1)
        obs[_OBS_POS_AGE] = pos_bars / max(time_stop_bars, 1)

        # Account state
        obs[_OBS_EQUITY_CURVE] = equity / max(starting_equity, 1.0)
        obs[_OBS_DRAWDOWN] = (peak_equity - equity) / max(peak_equity, 1.0)
        if recent_wins:
            obs[_OBS_WIN_RATE] = sum(recent_wins[-20:]) / len(recent_wins[-20:])

        # Killzone
        try:
            ts = pd.Timestamp(row["time"])
            hour = ts.hour
            obs[_OBS_KZ_ASIA] = 1.0 if 0 <= hour < 8 else 0.0
            obs[_OBS_KZ_LONDON] = 1.0 if 7 <= hour < 16 else 0.0
            obs[_OBS_KZ_NY] = 1.0 if 12 <= hour < 21 else 0.0
            obs[_OBS_HOUR_SIN] = np.sin(2 * np.pi * hour / 24.0)
            obs[_OBS_HOUR_COS] = np.cos(2 * np.pi * hour / 24.0)
        except Exception:
            pass

        return obs

    def predict_action(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        """Predict action from observation.

        Returns:
            (action_type, full_action_array)
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action[0]), action

    def should_skip(self, obs: np.ndarray, signal_direction: int) -> bool:
        """Decide whether to skip a signal based on RL model prediction.

        Args:
            obs: Current observation vector
            signal_direction: +1 for long, -1 for short

        Returns:
            True if the signal should be SKIPPED (agent says HOLD/CLOSE)
        """
        action_type, _ = self.predict_action(obs)

        # If agent says HOLD or CLOSE, skip the signal
        if action_type == ACT_HOLD or action_type == ACT_CLOSE:
            return True

        # If agent says enter opposite direction, skip
        if signal_direction == 1 and action_type != ACT_ENTER_LONG:
            return True
        if signal_direction == -1 and action_type != ACT_ENTER_SHORT:
            return True

        return False

    def get_exit_action(self, obs: np.ndarray) -> int:
        """Get exit action for an open position.

        Returns:
            ACT_HOLD (keep position) or ACT_CLOSE (close position)
        """
        action_type, _ = self.predict_action(obs)
        if action_type == ACT_CLOSE:
            return ACT_CLOSE
        return ACT_HOLD


# Need pandas for timestamp parsing
import pandas as pd
