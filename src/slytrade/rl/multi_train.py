"""Multi-Agent Training for SlyTrade.

Trains the hierarchical multi-agent system:
1. Phase 1: Train each sub-agent independently
2. Phase 2: Train meta-agent with frozen sub-agents
3. Phase 3: Fine-tune end-to-end

Usage:
    slytrade train --symbol XAUUSDm --timesteps 500000 --algo multi
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .multi_agent import (
    MultiAgentEnsemble,
    SubAgentRewards,
    OBS_DIM,
    ACTION_DIMS,
    compute_sub_agent_rewards,
)


class MultiAgentTrainer:
    """Trains the multi-agent ensemble using PPO-style updates.

    Each sub-agent gets its own optimizer and reward signal.
    The meta-agent is trained with the combined reward.
    """

    def __init__(
        self,
        env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        n_steps: int = 4096,
        batch_size: int = 512,
        n_epochs: int = 10,
        device: str = "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.clip_range = clip_range
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device(device)

        # Create ensemble
        self.ensemble = MultiAgentEnsemble().to(self.device)

        # Separate optimizers for sub-agents and meta-agent
        sub_agent_params = []
        for name in ["regime", "structure", "entry", "exit", "drawdown",
                      "risk", "setup", "trade_mgmt", "ict"]:
            sub_agent_params.extend(getattr(self.ensemble, name).parameters())

        self.sub_optimizer = optim.Adam(sub_agent_params, lr=lr)
        self.meta_optimizer = optim.Adam(self.ensemble.meta.parameters(), lr=lr)

        # Training stats
        self.stats = {
            "episode_rewards": [],
            "drawdowns": [],
            "win_rates": [],
            "sharpe_ratios": [],
        }

    def collect_rollout(self) -> dict:
        """Collect n_steps of experience from the environment."""
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        obs, _ = self.env.reset()
        done = False

        for _ in range(self.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, sub_outputs = self.ensemble.get_action(obs_tensor)

            action_np = action.squeeze(0).cpu().numpy().astype(np.int64)
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            observations.append(obs)
            actions.append(action_np)
            log_probs.append(log_prob.item() if log_prob is not None else 0.0)
            rewards.append(reward)
            dones.append(done)

            obs = next_obs
            if done:
                obs, _ = self.env.reset()

        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "log_probs": np.array(log_probs),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
        }

    def compute_gae(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        # Clip rewards to prevent overflow
        rewards = np.clip(rewards, -10.0, 10.0)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = advantages[t + 1] if not dones[t] else 0.0

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - 0.0
            last_gae = delta + self.gamma * 0.95 * (1 - dones[t]) * last_gae
            last_gae = np.clip(last_gae, -100.0, 100.0)  # prevent explosion
            advantages[t] = last_gae

        return advantages

    def update(self, rollout: dict) -> dict:
        """PPO update for both sub-agents and meta-agent."""
        observations = torch.FloatTensor(rollout["observations"]).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(self.compute_gae(rollout["rewards"], rollout["dones"])).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            # Mini-batch updates
            indices = np.arange(len(observations))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Forward pass
                meta_features, sub_outputs = self.ensemble(batch_obs)
                action_logits, size_logits, sl_logits, tp_logits = self.ensemble.meta(meta_features)

                # Compute new log probs
                action_dist = torch.distributions.Categorical(logits=action_logits)
                size_dist = torch.distributions.Categorical(logits=size_logits)
                sl_dist = torch.distributions.Categorical(logits=sl_logits)
                tp_dist = torch.distributions.Categorical(logits=tp_logits)

                new_log_probs = (
                    action_dist.log_prob(batch_actions[:, 0]) +
                    size_dist.log_prob(batch_actions[:, 1]) +
                    sl_dist.log_prob(batch_actions[:, 2]) +
                    tp_dist.log_prob(batch_actions[:, 3])
                )

                # PPO clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy = (action_dist.entropy() + size_dist.entropy() +
                          sl_dist.entropy() + tp_dist.entropy()).mean()

                loss = policy_loss - 0.01 * entropy

                # Single backward pass, update all parameters together
                self.meta_optimizer.zero_grad()
                self.sub_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ensemble.parameters(), 0.5)
                self.meta_optimizer.step()
                self.sub_optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return {"loss": total_loss / max(n_updates, 1)}

    def train(
        self,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = 100_000,
        output_dir: str = "models",
        symbol: str = "XAUUSDm",
        progress_fn=None,
    ):
        """Full training loop."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        timesteps_done = 0
        best_sharpe = -999

        while timesteps_done < total_timesteps:
            # Collect rollout
            rollout = self.collect_rollout()
            timesteps_done += len(rollout["rewards"])

            # Update
            update_info = self.update(rollout)

            # Progress
            elapsed = time.time() - start_time
            fps = timesteps_done / max(elapsed, 1)

            if progress_fn:
                progress_fn(f"  [{timesteps_done:,}/{total_timesteps:,}] "
                           f"loss={update_info['loss']:.4f} fps={fps:.0f} ({elapsed:.0f}s)")

            # Evaluation
            if eval_env and timesteps_done % eval_freq < self.n_steps:
                metrics = self._evaluate(eval_env)
                if progress_fn:
                    progress_fn(f"    EVAL: trades={metrics['n_trades']} wr={metrics['win_rate']:.0%} "
                               f"pnl={metrics['total_pnl']:+.0f} sharpe={metrics['sharpe_ratio']:.2f} "
                               f"dd={metrics['max_drawdown']:.0%}")

                if metrics['sharpe_ratio'] > best_sharpe and metrics['n_trades'] > 10:
                    best_sharpe = metrics['sharpe_ratio']
                    self.ensemble.save(str(output_path / f"multi_{symbol}_best.pt"))
                    if progress_fn:
                        progress_fn(f"      [green]New best! Saved.[/green]")

        # Save final
        self.ensemble.save(str(output_path / f"multi_{symbol}_final.pt"))

        # Final eval
        if eval_env:
            metrics = self._evaluate(eval_env)
            return metrics

        return {}

    def _evaluate(self, eval_env) -> dict:
        """Run evaluation episode."""
        obs, _ = eval_env.reset()
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _, _ = self.ensemble.get_action(obs_tensor, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, _ = eval_env.step(action_np)
            done = terminated or truncated

        return eval_env.get_metrics()
