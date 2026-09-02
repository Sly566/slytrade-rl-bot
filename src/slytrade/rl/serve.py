"""Model serving — load trained RL models and use them in live trading.

Supports both SB3 models (PPO/SAC/A2C) and MultiAgentEnsemble.

Usage in live trader:
    from slytrade.rl.serve import RLFilter, MultiAgentFilter

    # Single-agent (SB3)
    rl_filter = RLFilter("models/ppo_XAUUSDm_final.zip")

    # Multi-agent
    ma_filter = MultiAgentFilter("models/multi_XAUUSDm_best.pt")
    explanation = ma_filter.explain(obs)
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
    """Load a trained SB3 RL model and use it to filter signals."""

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

        Mirrors SlyTradeEnv._get_obs() — reads from aligned row dict/Series.
        All 90 features populated.
        """
        from .env import _safe_from_row
        obs = _safe_from_row(row, pos_dir, pos_entry, pos_risk, pos_bars,
                             equity, peak_equity, starting_equity,
                             recent_wins, time_stop_bars)
        return obs

    def predict_action(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        """Predict action from observation."""
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action[0]), action

    def should_skip(self, obs: np.ndarray, signal_direction: int) -> bool:
        """Decide whether to skip a signal based on RL model prediction."""
        action_type, _ = self.predict_action(obs)
        if action_type == ACT_HOLD or action_type == ACT_CLOSE:
            return True
        if signal_direction == 1 and action_type != ACT_ENTER_LONG:
            return True
        if signal_direction == -1 and action_type != ACT_ENTER_SHORT:
            return True
        return False

    def get_exit_action(self, obs: np.ndarray) -> int:
        """Get exit action for an open position."""
        action_type, _ = self.predict_action(obs)
        if action_type == ACT_CLOSE:
            return ACT_CLOSE
        return ACT_HOLD


class MultiAgentFilter:
    """Load a trained MultiAgentEnsemble and use it in live trading.

    Provides:
    - Signal filtering (take/skip)
    - Exit decisions
    - Full explainability (why each decision was made)
    - Sub-agent probability distributions
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        from .multi_agent import MultiAgentEnsemble

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.device = torch.device(device)
        self.ensemble = MultiAgentEnsemble().to(self.device)
        self.ensemble.load(str(path))
        self.ensemble.eval()
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
        """Build observation vector — mirrors env._get_obs()."""
        from .env import _safe_from_row
        return _safe_from_row(row, pos_dir, pos_entry, pos_risk, pos_bars,
                              equity, peak_equity, starting_equity,
                              recent_wins, time_stop_bars)

    def predict_action(self, obs: np.ndarray) -> tuple[np.ndarray, dict]:
        """Predict action and get explanation.

        Returns:
            action: [action_type, size_level, sl_mult, tp_mult]
            explanation: dict with reasoning, labels, confidence scores
        """
        import torch
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _, _ = self.ensemble.get_action(obs_tensor, deterministic=True)
            explanation = self.ensemble.explain(obs_tensor)
        return action.squeeze(0).cpu().numpy(), explanation

    def should_skip(self, obs: np.ndarray, signal_direction: int) -> tuple[bool, dict]:
        """Decide whether to skip a signal.

        Returns:
            (should_skip, explanation)
        """
        action, explanation = self.predict_action(obs)
        action_type = int(action[0])

        skip = False
        if action_type == ACT_HOLD or action_type == ACT_CLOSE:
            skip = True
        elif signal_direction == 1 and action_type != ACT_ENTER_LONG:
            skip = True
        elif signal_direction == -1 and action_type != ACT_ENTER_SHORT:
            skip = True

        return skip, explanation

    def get_exit_action(self, obs: np.ndarray) -> tuple[int, dict]:
        """Get exit action for an open position.

        Returns:
            (action, explanation)
        """
        action, explanation = self.predict_action(obs)
        action_type = int(action[0])
        if action_type == ACT_CLOSE:
            return ACT_CLOSE, explanation
        return ACT_HOLD, explanation

    def explain(self, obs: np.ndarray) -> dict:
        """Get human-readable explanation of current state."""
        import torch
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.ensemble.explain(obs_tensor)

    def get_sub_agent_probs(self, obs: np.ndarray) -> dict:
        """Get probability distributions from all sub-agents."""
        import torch
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.ensemble.get_sub_agent_probs(obs_tensor)
