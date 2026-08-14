"""Walk-forward validation and PPO training for the SlyTrade RL bot.

Walk-forward is the honest measurement of an edge: the model is trained only on
data strictly before each validation window, with an embargo gap so that
autocorrelated bars near the boundary cannot leak. This module also provides
the PPO training loop (wrapping stable-baselines3) and an Optuna hyperparameter
sweep. All SB3/optuna imports are lazy so the core package keeps working
without the `rl` extras installed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from slytrade.rl.environment import SlyTradeRLEnvironment

if TYPE_CHECKING:
    from slytrade.rl.dataset import RLDataset

EPS = 1e-12


@dataclass(frozen=True)
class WalkForwardFold:
    index: int
    train_start: int
    train_end: int  # exclusive
    val_start: int
    val_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive

    @property
    def train_range(self) -> tuple[int, int]:
        return self.train_start, self.train_end

    @property
    def val_range(self) -> tuple[int, int]:
        return self.val_start, self.val_end

    @property
    def test_range(self) -> tuple[int, int]:
        return self.test_start, self.test_end


def make_walk_forward_folds(
    total: int,
    *,
    train_window: int = 200_000,
    validation_window: int = 50_000,
    test_window: int = 50_000,
    embargo: int = 500,
    step: int | None = None,
) -> list[WalkForwardFold]:
    """Split a dataset into sequential walk-forward folds.

    Order within a fold: train -> embargo -> validation -> test (embargo also
    applied between validation and test). The next fold starts `step` bars
    after the previous one (defaults to the test window length).
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if train_window <= 0 or validation_window <= 0 or test_window <= 0:
        raise ValueError("windows must be positive")
    if embargo < 0:
        raise ValueError("embargo cannot be negative")

    step = step or test_window
    folds: list[WalkForwardFold] = []
    index = 0
    fold_index = 0
    while index + train_window + embargo + validation_window + embargo + test_window <= total:
        train_start = index
        train_end = train_start + train_window
        val_start = train_end + embargo
        val_end = val_start + validation_window
        test_start = val_end + embargo
        test_end = test_start + test_window
        if test_end > total:
            break
        folds.append(
            WalkForwardFold(
                index=fold_index,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_index += 1
        index += step
    if not folds:
        raise ValueError(
            "dataset too small for a single walk-forward fold with the given windows; "
            f"need at least {train_window + validation_window + test_window + 2 * embargo} bars, have {total}"
        )
    return folds


# ---------------------------------------------------------------------------
# Training (stable-baselines3, lazy import)
# ---------------------------------------------------------------------------

def train_ppo(
    env: SlyTradeRLEnvironment,
    *,
    total_timesteps: int = 100_000,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    policy_kwargs: dict | None = None,
    policy_type: str = "mlp",
    model_dir: str | None = None,
    verbose: int = 0,
):
    """Train a PPO policy on the environment. Returns the trained model.

    ``policy_type`` selects the network: "mlp" (default) or "lstm" (recurrent,
    for regime memory). All stable-baselines3 imports happen here so importing
    this module never requires torch/SB3 to be installed.
    """
    from stable_baselines3 import PPO

    policy = _policy_class(policy_type)
    model = PPO(
        policy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        seed=seed,
        policy_kwargs=policy_kwargs or {},
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps)
    if model_dir:
        model.save(model_dir)
    return model


def _policy_class(policy_type: str) -> str:
    normalized = policy_type.strip().lower()
    if normalized in ("mlp", "mlppolicy"):
        return "MlpPolicy"
    if normalized in ("lstm", "recurrent", "mlplstm"):
        return "MlpLstmPolicy"
    raise ValueError(f"unsupported policy_type {policy_type!r}; use 'mlp' or 'lstm'")


SUPPORTED_ALGORITHMS: tuple[str, ...] = ("ppo", "sac", "td3")


def resolve_algorithm(algorithm: str) -> str:
    """Normalise and validate an RL algorithm name (fail-fast, no imports)."""
    normalized = algorithm.strip().lower()
    if normalized not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported algorithm {algorithm!r}; choose from {SUPPORTED_ALGORITHMS}")
    return normalized


def _sb3_class(algorithm: str):
    """Lazily resolve the stable-baselines3 class for a supported algorithm."""
    normalized = resolve_algorithm(algorithm)
    if normalized == "ppo":
        from stable_baselines3 import PPO

        return PPO
    if normalized == "sac":
        from stable_baselines3 import SAC

        return SAC
    from stable_baselines3 import TD3

    return TD3


def train_policy(
    algorithm: str,
    env: SlyTradeRLEnvironment,
    *,
    total_timesteps: int = 100_000,
    seed: int = 42,
    model_dir: str | None = None,
    policy_kwargs: dict | None = None,
    policy_type: str = "mlp",
    verbose: int = 0,
):
    """Train a policy with any supported algorithm (PPO/SAC/TD3).

    All stable-baselines3 imports are lazy so the core package keeps working
    without the `rl` extras. Algorithm-specific hyperparameters are passed via
    ``policy_kwargs`` (the caller decides what makes sense for the algorithm).
    """
    cls = _sb3_class(algorithm)
    policy = _policy_class(policy_type) if algorithm == "ppo" else "MlpPolicy"
    model = cls(policy, env, seed=seed, policy_kwargs=policy_kwargs or {}, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)
    if model_dir:
        model.save(model_dir)
    return model


def evaluate_policy(
    model,
    env: SlyTradeRLEnvironment,
    *,
    episodes: int = 5,
    seed: int = 7,
) -> dict:
    """Run ``episodes`` episodes with a trained policy and return summary stats."""
    return evaluate_ppo(model, env, episodes=episodes, seed=seed)


def evaluate_ppo(
    model,
    env: SlyTradeRLEnvironment,
    *,
    episodes: int = 5,
    seed: int = 7,
) -> dict:
    """Run `episodes` complete episodes with the policy and return summary stats.

    Returns dict: total_return, mean_return_per_step, n_trades, final_equity,
    max_drawdown, win_rate (per trade, when a ledger with exits is available).
    """
    returns: list[float] = []
    trades: list[int] = []
    final_equities: list[float] = []
    max_drawdowns: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        equity_curve = [env.config.initial_balance]
        done = False
        truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            episode_return += reward
            equity_curve.append(info["equity"])
        returns.append(episode_return)
        trades.append(int(info["n_trades"]))
        final_equities.append(float(info["equity"]))
        max_drawdowns.append(_max_drawdown(equity_curve))

    return {
        "episodes": episodes,
        "mean_total_return": float(np.mean(returns)),
        "std_total_return": float(np.std(returns)),
        "mean_n_trades": float(np.mean(trades)),
        "mean_final_equity": float(np.mean(final_equities)),
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
    }


def _max_drawdown(equity_curve: list[float]) -> float:
    peaks = np.maximum.accumulate(equity_curve)
    return float(np.max((peaks - equity_curve) / np.maximum(peaks, EPS)))


# ---------------------------------------------------------------------------
# Optuna sweep (lazy import)
# ---------------------------------------------------------------------------

def optimize_ppo(
    env_factory: Callable[[int], SlyTradeRLEnvironment],
    *,
    n_trials: int = 20,
    total_timesteps_per_trial: int = 20_000,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Run an Optuna hyperparameter sweep for PPO on the trading environment.

    env_factory(seed) must return a fresh environment (same dataset scope, new
    seed). Returns (best_params, best_results).
    """
    import optuna

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        n_epochs = trial.suggest_int("n_epochs", 3, 20)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        gae = trial.suggest_float("gae_lambda", 0.9, 0.99)
        trial_seed = seed + trial.number

        env = env_factory(trial_seed)
        model = train_ppo(
            env,
            total_timesteps=total_timesteps_per_trial,
            seed=trial_seed,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae,
        )
        results = evaluate_ppo(model, env, episodes=3, seed=trial_seed)
        # Objective: maximize risk-adjusted return (mean return minus drawdown penalty)
        return float(results["mean_total_return"] - 0.5 * results["mean_max_drawdown"])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_seed = seed + study.best_trial.number
    best_env = env_factory(best_seed)
    best_model = train_ppo(best_env, total_timesteps=total_timesteps_per_trial, seed=best_seed, **best_params)
    best_results = evaluate_ppo(best_model, best_env, episodes=5, seed=best_seed)
    return best_params, best_results


def walk_forward_validation(
    dataset: RLDataset,
    folds: list[WalkForwardFold],
    *,
    total_timesteps: int = 20_000,
    seed: int = 42,
    reward_type: str = "raw",
    policy_type: str = "mlp",
) -> pd.DataFrame:
    """Train PPO on each fold's training window, evaluate on its test window.

    The scaler is re-fitted on each fold's train slice (no leakage). The
    ``reward_type`` is applied to BOTH the training and the test environments so
    the out-of-sample metric measures the same objective the policy optimised.
    Returns a DataFrame with one row per fold plus an AGGREGATE summary row.
    """
    from slytrade.rl.environment import RLEnvironmentConfig

    rows: list[dict] = []
    env_config = RLEnvironmentConfig(seed=seed, reward_type=reward_type)

    for fold in folds:
        # Fit the scaler on this fold's train window ONLY.
        scaler_params = dataset.fit_scaler(fold.train_start, fold.train_end)
        train_env = dataset.env_factory(
            fold.train_start,
            fold.train_end,
            seed=seed + fold.index,
            scaler_params=scaler_params,
            config=env_config,
        )
        model = train_ppo(train_env, total_timesteps=total_timesteps, seed=seed + fold.index, policy_type=policy_type)

        test_env = dataset.env_factory(
            fold.test_start,
            fold.test_end,
            seed=seed + fold.index,
            scaler_params=scaler_params,
            config=env_config,
        )
        results = evaluate_ppo(model, test_env, episodes=3, seed=seed + fold.index)

        row = {
            "fold": fold.index,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "test_mean_total_return": results["mean_total_return"],
            "test_mean_n_trades": results["mean_n_trades"],
            "test_mean_max_drawdown": results["mean_max_drawdown"],
        }
        rows.append(row)

    frame = pd.DataFrame(rows)

    # Aggregate summary row (aggregate out-of-sample performance)
    summary = {
        "fold": "AGGREGATE",
        "train_start": int(frame["train_start"].min()),
        "train_end": int(frame["test_end"].max()),
        "test_start": int(frame["test_start"].min()),
        "test_end": int(frame["test_end"].max()),
        "test_mean_total_return": float(frame["test_mean_total_return"].mean()),
        "test_mean_n_trades": float(frame["test_mean_n_trades"].mean()),
        "test_mean_max_drawdown": float(frame["test_mean_max_drawdown"].max()),
    }
    frame = pd.concat([frame, pd.DataFrame([summary])], ignore_index=True)
    return frame
