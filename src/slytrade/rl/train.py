"""Training loop for SlyTrade RL agent.

Uses Stable-Baselines3 PPO/SAC with Optuna hyperparameter tuning.

Usage:
    # Train PPO agent on XAUUSDm data
    python -m slytrade.rl.train --symbol XAUUSDm --data data/aligned/XAUUSDm.parquet

    # Train with Optuna hyperparameter search
    python -m slytrade.rl.train --symbol XAUUSDm --data data/aligned/XAUUSDm.parquet --tune

    # Resume training from checkpoint
    python -m slytrade.rl.train --symbol XAUUSDm --data data/aligned/XAUUSDm.parquet --resume models/ppo_slytrade_100000.zip
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO, SAC, A2C
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import optuna
    from optuna.integration import SB3PruningCallback
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from ..backtest.specs import AccountSpec, spec_for_symbol
from ..data.features import DEFAULT_CONFIG, process_bars
from ..data.mtf_align import _asof_merge, _prep_htf_frame
from ..data.time import timeframe_timedelta
from ..strategy.config import rl_training_persona
from .env import SlyTradeEnv


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aligned_data(data_path: str, symbol: str = "XAUUSDm") -> pd.DataFrame:
    """Load pre-aligned M1 data with HTF features.

    If data_path is a parquet file, load it directly.
    If it's a directory, look for {symbol}_aligned.parquet.
    """
    path = Path(data_path)
    if path.is_file():
        return pd.read_parquet(path)
    elif path.is_dir():
        candidates = list(path.glob(f"*{symbol}*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No parquet files matching {symbol} in {path}")
        return pd.read_parquet(sorted(candidates)[-1])  # latest
    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")


def prepare_training_data(
    raw_m1: pd.DataFrame,
    htf_frames: dict[str, pd.DataFrame] | None = None,
    symbol: str = "XAUUSDm",
) -> pd.DataFrame:
    """Process raw M1 bars and align HTF features for training.

    This replicates the same pipeline as the live trader:
    1. Compute M1 features (ATR, structure flags, etc.)
    2. Fetch/compute HTF features (M5, M15, M30, H1)
    3. Causal asof merge
    """
    from ..data.mtf_align import _asof_merge, _prep_htf_frame

    m1 = process_bars(raw_m1, "M1", DEFAULT_CONFIG)

    if htf_frames is None:
        # If no HTF data provided, we'll work with M1-only features
        return m1

    # Align HTF features onto M1
    df = m1.copy().sort_values("time").reset_index(drop=True)
    for tf, htf in htf_frames.items():
        if htf.empty:
            continue
        dur = timeframe_timedelta(tf)
        htf = htf.copy()
        htf["decision_time"] = htf["time"] + dur
        prepped = _prep_htf_frame(htf, tf)
        df = _asof_merge(df, prepped, tf)

    return df


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class MetricsCallback(BaseCallback):
    """Log training metrics every N steps."""

    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._episode_rewards = []
        self._episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode stats when episode ends
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self._episode_rewards.append(ep["r"])
                self._episode_lengths.append(ep["l"])

        if self.n_calls % self.log_interval == 0 and self._episode_rewards:
            recent = self._episode_rewards[-100:]
            print(f"  step={self.n_calls} "
                  f"episodes={len(self._episode_rewards)} "
                  f"avg_reward={np.mean(recent):.3f} "
                  f"avg_length={np.mean(self._episode_lengths[-100:]):.0f} "
                  f"best={max(recent):.3f} "
                  f"worst={min(recent):.3f}")
        return True


class TradeMetricsCallback(BaseCallback):
    """Log trade-level metrics from environment info."""

    def __init__(self, log_interval: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._total_trades = 0
        self._total_wins = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("trade_closed"):
                self._total_trades += 1
                if info.get("close_pnl", 0) > 0:
                    self._total_wins += 1

        if self.n_calls % self.log_interval == 0 and self._total_trades > 0:
            wr = self._total_wins / self._total_trades
            print(f"  [TRADES] total={self._total_trades} wins={self._total_wins} "
                  f"win_rate={wr:.1%}")
        return True


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def create_optuna_study(
    aligned_df: pd.DataFrame,
    n_trials: int = 50,
    n_timesteps: int = 100_000,
    output_dir: str = "models/optuna",
) -> dict:
    """Run Optuna hyperparameter search for PPO.

    Returns best hyperparameters.
    """
    if not HAS_OPTUNA:
        raise ImportError("optuna not installed. Run: pip install optuna")

    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        n_epochs = trial.suggest_int("n_epochs", 3, 20)
        net_arch = trial.suggest_categorical("net_arch", [
            [64, 64], [128, 128], [256, 256], [128, 64], [256, 128, 64],
        ])

        # Create environment
        def make_env():
            env = SlyTradeEnv(aligned_df, max_bars=5000)
            env = Monitor(env)
            return env

        vec_env = DummyVecEnv([make_env])

        # Create model
        policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, n_steps=n_steps,
            batch_size=batch_size, n_epochs=n_epochs,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=42,
        )

        # Train
        model.learn(total_timesteps=n_timesteps)

        # Evaluate
        eval_env = SlyTradeEnv(aligned_df, max_bars=10000)
        obs, info = eval_env.reset()
        total_reward = 0.0
        for _ in range(10000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        metrics = eval_env.get_metrics()
        # Optimize for Sharpe ratio (risk-adjusted returns)
        sharpe = metrics.get("sharpe_ratio", 0.0)
        win_rate = metrics.get("win_rate", 0.0)
        max_dd = metrics.get("max_drawdown", 1.0)

        # Combined objective: Sharpe + win_rate - drawdown_penalty
        objective_val = sharpe + win_rate * 2.0 - max_dd * 5.0

        return objective_val

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    best = study.best_params
    with open(f"{output_dir}/best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    print(f"\nBest trial: {study.best_trial.value:.3f}")
    print(f"Best params: {json.dumps(best, indent=2)}")

    return best


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_agent(
    data_path: str,
    symbol: str = "XAUUSDm",
    algo: str = "ppo",
    total_timesteps: int = 500_000,
    output_dir: str = "models",
    *,
    tune: bool = False,
    resume: str | None = None,
    n_envs: int = 1,
    max_bars_per_episode: int = 5000,
    eval_interval: int = 50_000,
    checkpoint_interval: int = 50_000,
    verbose: int = 1,
) -> str:
    """Train an RL agent on historical data.

    Args:
        data_path: Path to aligned data (parquet)
        symbol: Trading symbol
        algo: Algorithm (ppo, sac, a2c)
        total_timesteps: Total training timesteps
        output_dir: Directory to save models
        tune: Run Optuna hyperparameter search first
        resume: Path to checkpoint to resume from
        n_envs: Number of parallel environments
        max_bars_per_episode: Max bars per episode
        eval_interval: Evaluate every N timesteps
        checkpoint_interval: Save checkpoint every N timesteps
        verbose: Verbosity level

    Returns:
        Path to trained model
    """
    if not HAS_SB3:
        raise ImportError(
            "stable-baselines3 not installed. "
            "Run: pip install 'slytrade-rl-bot[rl]'"
        )

    print(f"Loading data from {data_path} ...")
    aligned_df = load_aligned_data(data_path, symbol)
    print(f"  Loaded {len(aligned_df)} bars from {aligned_df['time'].min()} to {aligned_df['time'].max()}")

    # Optuna tuning
    best_params = {}
    if tune:
        print("\nRunning Optuna hyperparameter search ...")
        best_params = create_optuna_study(
            aligned_df, n_trials=50, n_timesteps=100_000,
            output_dir=f"{output_dir}/optuna",
        )

    # Create environments
    def make_env():
        env = SlyTradeEnv(aligned_df, max_bars=max_bars_per_episode)
        env = Monitor(env)
        return env

    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env])

    eval_env = DummyVecEnv([make_env])

    # Create model
    algo_cls = {"ppo": PPO, "sac": SAC, "a2c": A2C}[algo.lower()]

    model_kwargs = {
        "policy": "MlpPolicy",
        "env": vec_env,
        "verbose": verbose,
        "seed": 42,
        "tensorboard_log": f"{output_dir}/tb_logs",
    }

    # Apply Optuna best params if available
    if best_params:
        model_kwargs.update(best_params)
        if "net_arch" in best_params:
            model_kwargs["policy_kwargs"] = dict(
                net_arch=dict(pi=best_params["net_arch"], vf=best_params["net_arch"])
            )
            del model_kwargs["net_arch"]

    if resume:
        print(f"\nResuming from {resume} ...")
        model = algo_cls.load(resume, env=vec_env, **{k: v for k, v in model_kwargs.items() if k not in ("policy", "env")})
    else:
        model = algo_cls(**model_kwargs)

    # Callbacks
    os.makedirs(output_dir, exist_ok=True)
    callbacks = [
        MetricsCallback(log_interval=1000),
        TradeMetricsCallback(log_interval=5000),
        CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=f"{output_dir}/checkpoints",
            name_prefix=f"{algo}_{symbol}",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"{output_dir}/best",
            eval_freq=eval_interval,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    # Train
    print(f"\nTraining {algo.upper()} for {total_timesteps:,} timesteps ...")
    print(f"  Max bars/episode: {max_bars_per_episode}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Output: {output_dir}")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = f"{output_dir}/{algo}_{symbol}_final.zip"
    model.save(final_path)
    print(f"\nModel saved to {final_path}")

    # Final evaluation
    print("\nFinal evaluation (10 episodes) ...")
    eval_env2 = SlyTradeEnv(aligned_df, max_bars=len(aligned_df))
    all_metrics = []
    for ep in range(10):
        obs, info = eval_env2.reset()
        total_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env2.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        metrics = eval_env2.get_metrics()
        metrics["episode_reward"] = total_reward
        all_metrics.append(metrics)
        print(f"  ep={ep} trades={metrics['n_trades']} wr={metrics['win_rate']:.0%} "
              f"pnl={metrics['total_pnl']:+.0f} sharpe={metrics['sharpe_ratio']:.2f} "
              f"max_dd={metrics['max_drawdown']:.1%}")

    # Average metrics
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f"\nAverage: trades={avg['n_trades']:.0f} wr={avg['win_rate']:.0%} "
          f"pnl={avg['total_pnl']:+.0f} sharpe={avg['sharpe_ratio']:.2f} "
          f"max_dd={avg['max_drawdown']:.1%}")

    # Save metrics
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump({"average": avg, "episodes": all_metrics}, f, indent=2, default=str)

    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train SlyTrade RL agent")
    ap.add_argument("--symbol", default="XAUUSDm")
    ap.add_argument("--data", required=True, help="Path to aligned data (parquet)")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac", "a2c"])
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--output", default="models")
    ap.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    ap.add_argument("--resume", help="Path to checkpoint to resume from")
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--max-bars", type=int, default=5000)
    args = ap.parse_args()

    train_agent(
        data_path=args.data,
        symbol=args.symbol,
        algo=args.algo,
        total_timesteps=args.timesteps,
        output_dir=args.output,
        tune=args.tune,
        resume=args.resume,
        n_envs=args.n_envs,
        max_bars_per_episode=args.max_bars,
    )


if __name__ == "__main__":
    main()
