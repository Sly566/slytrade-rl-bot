"""Multi-Agent RL System for SlyTrade.

Architecture: Hierarchical Multi-Agent with specialized sub-agents.

Each sub-agent is a dedicated neural network trained on its specific concern.
A meta-agent (orchestrator) learns to combine sub-agent recommendations
into final trading decisions.

Sub-Agents:
1. RegimeDetector    — identifies market regime (trending/ranging/volatile/breakout)
2. StructureAgent    — interprets BOS/CHoCH/sweeps for directional bias
3. SniperEntry       — precision entry timing (OB bounces, FVG fills, sweep reversals)
4. OptimumExit       — exit optimization (trailing, partial, time-based)
5. DrawdownControl   — monitors and controls drawdown in real-time
6. RiskManager       — position sizing based on account/volatility/regime
7. SetupGrader       — grades setup quality (A/B/C) for filtering
8. TradeManager      — manages open positions (BE moves, scale in/out)
9. ICTPersona        — applies institutional behavior patterns (killzones, liquidity)

Meta-Agent:
- Takes all sub-agent outputs as input
- Produces final action: [action_type, size_level, sl_mult, tp_mult]
- Learns to weight sub-agents by regime and context

Training Strategy:
- Phase 1: Train each sub-agent independently on its specific task
- Phase 2: Freeze sub-agents, train meta-agent to combine them
- Phase 3: Fine-tune end-to-end with shared gradients
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBS_DIM = 39  # from env.py

# Sub-agent output dimensions
REGIME_DIM = 4      # trending_up, trending_down, ranging, volatile
STRUCTURE_DIM = 4   # bull_bias, bear_bias, bos_strength, choch_strength
ENTRY_DIM = 4       # long_signal, short_signal, confidence, setup_grade
EXIT_DIM = 3        # hold, take_profit, cut_loss
DRAWDOWN_DIM = 3    # normal, warning, critical (reduces size/exposure)
RISK_DIM = 3        # lot_size_idx (0.01, 0.04, 0.08)
SETUP_DIM = 3       # grade_A, grade_B, grade_C
TRADE_MGMT_DIM = 4  # hold, move_sl_be, partial_close, add_position
ICT_DIM = 4         # in_killzone, liquidity_sweep, displacement, fvg_fill

# Meta-agent input = sum of all sub-agent outputs
META_INPUT_DIM = (REGIME_DIM + STRUCTURE_DIM + ENTRY_DIM + EXIT_DIM +
                  DRAWDOWN_DIM + RISK_DIM + SETUP_DIM + TRADE_MGMT_DIM + ICT_DIM)

# Final action dimensions (same as single-agent)
ACTION_DIMS = [4, 3, 4, 5]  # action_type, size_level, sl_mult, tp_mult


class Regime(IntEnum):
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    VOLATILE = 3


# ---------------------------------------------------------------------------
# Sub-Agent Networks
# ---------------------------------------------------------------------------
class SubAgent(nn.Module):
    """Base class for all sub-agents."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self.forward(x), dim=-1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (for meta-agent input)."""
        return self.forward(x)


class RegimeDetector(SubAgent):
    """Identifies market regime from structure + volatility features.

    Input: M1/M5/M15 structure flags, ATR, displacement, BOS/CHoCH
    Output: [trending_up, trending_down, ranging, volatile]
    """
    def __init__(self):
        super().__init__(OBS_DIM, REGIME_DIM, hidden=128)


class StructureAgent(SubAgent):
    """Interprets market structure for directional bias.

    Input: BOS/CHoCH flags across M1/M5/M15, swing highs/lows
    Output: [bull_bias, bear_bias, bos_strength, choch_strength]
    """
    def __init__(self):
        super().__init__(OBS_DIM, STRUCTURE_DIM, hidden=64)


class SniperEntry(SubAgent):
    """Precision entry timing — identifies high-probability entry zones.

    Input: OB proximity, FVG proximity, sweep signals, displacement,
           killzone status, structure alignment
    Output: [long_signal, short_signal, confidence, setup_grade]
    """
    def __init__(self):
        super().__init__(OBS_DIM, ENTRY_DIM, hidden=128)


class OptimumExit(SubAgent):
    """Exit optimization — when to take profit or cut losses.

    Input: position state, R-multiple, time in trade, ATR expansion,
           opposing structure signals
    Output: [hold, take_profit, cut_loss]
    """
    def __init__(self):
        super().__init__(OBS_DIM, EXIT_DIM, hidden=64)


class DrawdownControl(SubAgent):
    """Real-time drawdown monitoring and control.

    This is the CRITICAL agent — it overrides all other agents when
    drawdown exceeds thresholds. It learns to:
    - Reduce position sizes as DD increases
    - Force-close positions at extreme DD
    - Prevent new entries during DD recovery
    - Scale back in as DD decreases

    Input: equity curve, drawdown, recent trade results, win rate
    Output: [normal, warning, critical]
    - normal: full position sizing allowed
    - warning: reduce size by 50%, tighter stops
    - critical: no new entries, close existing positions
    """
    def __init__(self):
        super().__init__(OBS_DIM, DRAWDOWN_DIM, hidden=64)


class RiskManager(SubAgent):
    """Position sizing based on account state and market conditions.

    Input: equity, drawdown, ATR, regime, win rate
    Output: [lot_size_idx] → 0.01, 0.04, 0.08 lots
    """
    def __init__(self):
        super().__init__(OBS_DIM, RISK_DIM, hidden=64)


class SetupGrader(SubAgent):
    """Grades setup quality for filtering.

    Input: structure alignment across TFs, displacement, killzone,
           OB/FVG quality, sweep confirmation
    Output: [grade_A, grade_B, grade_C]
    - A grade: full size, wider TP
    - B grade: standard size, standard TP
    - C grade: reduced size, tighter TP, trailing stop
    """
    def __init__(self):
        super().__init__(OBS_DIM, SETUP_DIM, hidden=64)


class TradeManager(SubAgent):
    """Manages open positions — BE moves, partial closes, scaling.

    Input: position state, R-multiple, time in trade, opposing signals
    Output: [hold, move_sl_be, partial_close, add_position]
    """
    def __init__(self):
        super().__init__(OBS_DIM, TRADE_MGMT_DIM, hidden=64)


class ICTPersona(SubAgent):
    """Applies ICT/SMC institutional behavior patterns.

    Input: killzone status, liquidity sweep, displacement, FVG fill,
           Judas swing, Silver Bullet, Power of 3
    Output: [in_killzone, liquidity_sweep, displacement, fvg_fill]
    """
    def __init__(self):
        super().__init__(OBS_DIM, ICT_DIM, hidden=64)


# ---------------------------------------------------------------------------
# Meta-Agent (Orchestrator)
# ---------------------------------------------------------------------------
class MetaAgent(nn.Module):
    """Combines sub-agent outputs into final trading decisions.

    Takes all sub-agent logits/features as input and produces:
    - action_type: ENTER_LONG, ENTER_SHORT, CLOSE, HOLD
    - size_level: 0.01, 0.04, 0.08 lots
    - sl_mult: 1.0x, 1.5x, 2.0x, 2.5x ATR
    - tp_mult: 0.5R, 1.0R, 1.5R, 2.0R, 2.5R

    The meta-agent learns to:
    - Weight sub-agents by context (e.g., trust DrawdownControl more in DD)
    - Resolve conflicts between sub-agents
    - Adapt to changing market conditions
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(META_INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Multi-head output (same as single-agent action space)
        self.action_head = nn.Linear(64, ACTION_DIMS[0])      # 4: enter_long, enter_short, close, hold
        self.size_head = nn.Linear(64, ACTION_DIMS[1])         # 3: 0.01, 0.04, 0.08
        self.sl_head = nn.Linear(64, ACTION_DIMS[2])           # 4: 1.0x, 1.5x, 2.0x, 2.5x
        self.tp_head = nn.Linear(64, ACTION_DIMS[3])           # 5: 0.5R, 1.0R, 1.5R, 2.0R, 2.5R
        # Value head (critic) — required for proper GAE advantage estimation
        self.value_head = nn.Linear(64, 1)

    def forward(self, sub_agent_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            sub_agent_features: concatenated sub-agent outputs [batch, META_INPUT_DIM]

        Returns:
            (action_logits, size_logits, sl_logits, tp_logits, value)
        """
        h = self.net(sub_agent_features)
        return (
            self.action_head(h),
            self.size_head(h),
            self.sl_head(h),
            self.tp_head(h),
            self.value_head(h).squeeze(-1),
        )

    def get_action(self, sub_agent_features: torch.Tensor, deterministic: bool = False):
        """Get action from sub-agent features.

        Returns:
            action: [action_type, size_level, sl_mult, tp_mult]
            log_prob: sum of log probabilities for all heads
            value: state value estimate (for GAE)
        """
        action_logits, size_logits, sl_logits, tp_logits, value = self.forward(sub_agent_features)

        if deterministic:
            action = torch.stack([
                action_logits.argmax(-1),
                size_logits.argmax(-1),
                sl_logits.argmax(-1),
                tp_logits.argmax(-1),
            ])
            return action, None, value

        # Sample from categorical distributions
        action_dist = torch.distributions.Categorical(logits=action_logits)
        size_dist = torch.distributions.Categorical(logits=size_logits)
        sl_dist = torch.distributions.Categorical(logits=sl_logits)
        tp_dist = torch.distributions.Categorical(logits=tp_logits)

        action = torch.stack([
            action_dist.sample(),
            size_dist.sample(),
            sl_dist.sample(),
            tp_dist.sample(),
        ])
        log_prob = (action_dist.log_prob(action[0]) +
                    size_dist.log_prob(action[1]) +
                    sl_dist.log_prob(action[2]) +
                    tp_dist.log_prob(action[3]))

        return action, log_prob, value


# ---------------------------------------------------------------------------
# Multi-Agent Ensemble
# ---------------------------------------------------------------------------
class MultiAgentEnsemble(nn.Module):
    """Full multi-agent system: 9 sub-agents + 1 meta-agent.

    Usage:
        ensemble = MultiAgentEnsemble()
        obs = torch.randn(1, OBS_DIM)
        action, log_prob, sub_outputs = ensemble.get_action(obs)
    """

    def __init__(self):
        super().__init__()
        self.regime = RegimeDetector()
        self.structure = StructureAgent()
        self.entry = SniperEntry()
        self.exit = OptimumExit()
        self.drawdown = DrawdownControl()
        self.risk = RiskManager()
        self.setup = SetupGrader()
        self.trade_mgmt = TradeManager()
        self.ict = ICTPersona()
        self.meta = MetaAgent()

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through all sub-agents + meta-agent.

        Args:
            obs: observation tensor [batch, OBS_DIM]

        Returns:
            meta_features: concatenated sub-agent outputs [batch, META_INPUT_DIM]
            sub_outputs: dict of sub-agent raw outputs
        """
        sub_outputs = {
            "regime": self.regime.get_features(obs),
            "structure": self.structure.get_features(obs),
            "entry": self.entry.get_features(obs),
            "exit": self.exit.get_features(obs),
            "drawdown": self.drawdown.get_features(obs),
            "risk": self.risk.get_features(obs),
            "setup": self.setup.get_features(obs),
            "trade_mgmt": self.trade_mgmt.get_features(obs),
            "ict": self.ict.get_features(obs),
        }
        meta_features = torch.cat(list(sub_outputs.values()), dim=-1)
        return meta_features, sub_outputs

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Get trading action from observation.

        Returns:
            action: [action_type, size_level, sl_mult, tp_mult]
            log_prob: sum log prob (for PPO)
            value: state value estimate (for GAE)
            sub_outputs: dict of sub-agent outputs (for analysis)
        """
        meta_features, sub_outputs = self.forward(obs)
        action, log_prob, value = self.meta.get_action(meta_features, deterministic)
        return action, log_prob, value, sub_outputs

    def get_sub_agent_probs(self, obs: torch.Tensor) -> dict[str, np.ndarray]:
        """Get probability distributions from each sub-agent (for analysis)."""
        with torch.no_grad():
            return {
                "regime": self.regime.get_probs(obs).numpy(),
                "structure": self.structure.get_probs(obs).numpy(),
                "entry": self.entry.get_probs(obs).numpy(),
                "exit": self.exit.get_probs(obs).numpy(),
                "drawdown": self.drawdown.get_probs(obs).numpy(),
                "risk": self.risk.get_probs(obs).numpy(),
                "setup": self.setup.get_probs(obs).numpy(),
                "trade_mgmt": self.trade_mgmt.get_probs(obs).numpy(),
                "ict": self.ict.get_probs(obs).numpy(),
            }

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, weights_only=True))


# ---------------------------------------------------------------------------
# Multi-Agent Environment Wrapper
# ---------------------------------------------------------------------------
class MultiAgentEnv:
    """Wraps SlyTradeEnv to use MultiAgentEnsemble for decisions.

    This is a drop-in replacement for the single-agent env that:
    1. Takes observations from SlyTradeEnv
    2. Runs them through the multi-agent ensemble
    3. Returns actions in the same format

    The sub-agents provide interpretability — you can see WHY the
    bot made each decision by inspecting sub-agent outputs.
    """

    def __init__(self, ensemble: MultiAgentEnsemble):
        self.ensemble = ensemble

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, dict]:
        """Predict action from observation.

        Returns:
            action: numpy array [action_type, size_level, sl_mult, tp_mult]
            info: dict with sub-agent outputs for interpretability
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action, _, sub_outputs = self.ensemble.get_action(obs_tensor, deterministic)

        info = {}
        for name, output in sub_outputs.items():
            info[name] = output.squeeze(0).numpy()

        return action.squeeze(0).cpu().numpy().astype(np.int64), info


# ---------------------------------------------------------------------------
# Sub-Agent Reward Shaping
# ---------------------------------------------------------------------------
@dataclass
class SubAgentRewards:
    """Individual reward signals for each sub-agent.

    Each sub-agent gets its own reward signal based on how well
    it performs its specific task. This allows targeted learning.
    """
    regime: float = 0.0       # reward for correct regime identification
    structure: float = 0.0    # reward for correct directional bias
    entry: float = 0.0        # reward for profitable entries
    exit: float = 0.0         # reward for optimal exits
    drawdown: float = 0.0     # reward for keeping DD low
    risk: float = 0.0         # reward for appropriate position sizing
    setup: float = 0.0        # reward for selecting quality setups
    trade_mgmt: float = 0.0   # reward for good position management
    ict: float = 0.0          # reward for ICT pattern recognition

    def to_array(self) -> np.ndarray:
        return np.array([
            self.regime, self.structure, self.entry, self.exit,
            self.drawdown, self.risk, self.setup, self.trade_mgmt, self.ict,
        ], dtype=np.float32)


def compute_sub_agent_rewards(
    trade_result: dict | None,
    drawdown: float,
    regime_correct: bool,
    in_killzone: bool,
    setup_grade: str,
) -> SubAgentRewards:
    """Compute individual rewards for each sub-agent based on trade outcome.

    This is called after each trade closes to provide targeted feedback
    to each sub-agent.
    """
    rewards = SubAgentRewards()

    # Drawdown control — always active, penalizes high DD
    if drawdown < 0.05:
        rewards.drawdown = 0.1  # reward for low DD
    elif drawdown < 0.10:
        rewards.drawdown = 0.0
    elif drawdown < 0.20:
        rewards.drawdown = -0.5
    else:
        rewards.drawdown = -2.0  # heavy penalty for high DD

    if trade_result is None:
        return rewards

    pnl = trade_result.get("pnl", 0.0)
    won = pnl > 0

    # Entry agent — reward for profitable entries
    rewards.entry = 0.5 if won else -0.3

    # Exit agent — reward for exits (winning = good exit timing)
    rewards.exit = 0.3 if won else -0.2

    # Risk manager — reward for appropriate sizing
    # (smaller losses = better risk management)
    if won:
        rewards.risk = 0.2
    else:
        rewards.risk = -0.1 if abs(pnl) < 100 else -0.5

    # Setup grader — A-grade setups should win more
    if setup_grade == "A":
        rewards.setup = 0.5 if won else -0.3
    elif setup_grade == "B":
        rewards.setup = 0.2 if won else -0.2
    else:
        rewards.setup = 0.1 if won else -0.1

    # ICT persona — reward for trading in killzones
    if in_killzone:
        rewards.ict = 0.2 if won else -0.1
    else:
        rewards.ict = -0.1 if won else 0.0  # slight penalty for trading outside KZ

    # Regime — reward for correct regime identification
    if regime_correct:
        rewards.regime = 0.1

    return rewards
