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
        values = []
        rewards = []
        dones = []

        obs, _ = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_rewards = []
        episode_lengths = []
        n_trades = 0
        wins = 0
        total_pnl = 0.0
        sub_agent_sums = {}  # running sum of sub-agent outputs

        for step in range(self.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value, sub_outputs = self.ensemble.get_action(obs_tensor)

            # Track sub-agent output magnitudes
            for name, out in sub_outputs.items():
                if name not in sub_agent_sums:
                    sub_agent_sums[name] = np.zeros(out.shape[-1])
                sub_agent_sums[name] += out.squeeze(0).cpu().numpy()

            action_np = action.squeeze(0).cpu().numpy().astype(np.int64)
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            observations.append(obs)
            actions.append(action_np)
            log_probs.append(log_prob.item() if log_prob is not None else 0.0)
            values.append(value.item() if value is not None else 0.0)
            rewards.append(reward)
            dones.append(done)

            episode_reward += reward
            episode_length += 1

            # Track trade metrics from info
            if info.get("trade_closed"):
                n_trades += 1
                pnl = info.get("close_pnl", 0.0)
                total_pnl += pnl
                if pnl > 0:
                    wins += 1

            obs = next_obs
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0.0
                episode_length = 0
                obs, _ = self.env.reset()

        # Compute rollout stats
        self._last_rollout_stats = {
            "mean_ep_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_ep_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "n_trades": n_trades,
            "win_rate": wins / max(n_trades, 1),
            "total_pnl": total_pnl,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "sub_agent_sums": sub_agent_sums,
        }

        return {
            "observations": np.array(observations),
            "actions": np.array(actions),
            "log_probs": np.array(log_probs),
            "values": np.array(values),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
        }

    def compute_gae(self, rewards: np.ndarray, dones: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation using value function.

        Returns:
            advantages: GAE advantages
            returns: discounted returns (targets for value function)
        """
        rewards = np.clip(rewards, -10.0, 10.0)
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1] if not dones[t] else 0.0

            # TD error: r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * 0.95 * (1 - dones[t]) * last_gae
            last_gae = np.clip(last_gae, -50.0, 50.0)
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]  # GAE target

        return advantages, returns

    def update(self, rollout: dict) -> dict:
        """PPO update for both sub-agents and meta-agent."""
        observations = torch.FloatTensor(rollout["observations"]).to(self.device)
        actions = torch.LongTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages_np, returns_np = self.compute_gae(
            rollout["rewards"], rollout["dones"], rollout["values"]
        )
        advantages = torch.FloatTensor(advantages_np).to(self.device)
        returns = torch.FloatTensor(returns_np).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = np.arange(len(observations))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass
                meta_features, sub_outputs = self.ensemble(batch_obs)
                action_logits, size_logits, sl_logits, tp_logits, values_pred = self.ensemble.meta(meta_features)

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

                # Value loss (MSE)
                value_loss = 0.5 * (values_pred - batch_returns).pow(2).mean()

                # Entropy bonus
                entropy = (action_dist.entropy() + size_dist.entropy() +
                          sl_dist.entropy() + tp_dist.entropy()).mean()

                # Combined loss: policy + 0.5 * value - 0.01 * entropy
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # Single backward pass, update all parameters together
                self.meta_optimizer.zero_grad()
                self.sub_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ensemble.parameters(), 0.5)
                self.meta_optimizer.step()
                self.sub_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

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

            # Progress — detailed stats
            elapsed = time.time() - start_time
            fps = timesteps_done / max(elapsed, 1)
            stats = self._last_rollout_stats

            if progress_fn:
                # Main metrics
                progress_fn(
                    f"  [{timesteps_done:,}/{total_timesteps:,}] "
                    f"pi_loss={update_info['policy_loss']:.4f} "
                    f"v_loss={update_info['value_loss']:.4f} "
                    f"entropy={update_info['entropy']:.3f} "
                    f"fps={fps:.0f} ({elapsed:.0f}s)"
                )
                # Trade metrics
                progress_fn(
                    f"    trades={stats['n_trades']} wr={stats['win_rate']:.0%} "
                    f"pnl={stats['total_pnl']:+.0f} "
                    f"ep_reward={stats['mean_ep_reward']:.1f} "
                    f"ep_len={stats['mean_ep_length']:.0f}"
                )
                # Reward distribution
                progress_fn(
                    f"    reward: mean={stats['mean_reward']:.3f} std={stats['std_reward']:.3f} "
                    f"max={stats['max_reward']:.2f} min={stats['min_reward']:.2f}"
                )
                # Sub-agent output summary
                sub = stats['sub_agent_sums']
                n = max(stats.get('n_trades', 1), 1)
                regime_avg = sub.get("regime", np.zeros(4)) / max(self.n_steps, 1)
                dd_avg = sub.get("drawdown", np.zeros(3)) / max(self.n_steps, 1)
                entry_avg = sub.get("entry", np.zeros(4)) / max(self.n_steps, 1)
                exit_avg = sub.get("exit", np.zeros(3)) / max(self.n_steps, 1)
                risk_avg = sub.get("risk", np.zeros(3)) / max(self.n_steps, 1)
                setup_avg = sub.get("setup", np.zeros(3)) / max(self.n_steps, 1)
                ict_avg = sub.get("ict", np.zeros(4)) / max(self.n_steps, 1)

                regime_names = ["trend_up", "trend_dn", "ranging", "volatile"]
                progress_fn(f"    regime: {regime_names[np.argmax(regime_avg)]} ({regime_avg[np.argmax(regime_avg)]:.2f})")
                progress_fn(f"    drawdown: normal={dd_avg[0]:.2f} warn={dd_avg[1]:.2f} critical={dd_avg[2]:.2f}")
                progress_fn(f"    entry: long={entry_avg[0]:.2f} short={entry_avg[1]:.2f} conf={entry_avg[2]:.2f}")
                progress_fn(f"    exit: hold={exit_avg[0]:.2f} tp={exit_avg[1]:.2f} cut={exit_avg[2]:.2f}")
                progress_fn(f"    risk: {risk_avg} setup: A={setup_avg[0]:.2f} B={setup_avg[1]:.2f} C={setup_avg[2]:.2f}")
                progress_fn(f"    ict: kz={ict_avg[0]:.2f} sweep={ict_avg[1]:.2f} disp={ict_avg[2]:.2f} fvg={ict_avg[3]:.2f}")

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
                action, _, _, _ = self.ensemble.get_action(obs_tensor, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()
            obs, _, terminated, truncated, _ = eval_env.step(action_np)
            done = terminated or truncated

        return eval_env.get_metrics()
