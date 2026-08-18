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
from typing import TYPE_CHECKING, Any

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


@dataclass(frozen=True)
class FoldWindows:
    train_window: int
    validation_window: int
    test_window: int
    embargo: int
    step: int | None = None


def resolve_fold_windows(
    total: int,
    *,
    train_window: int = 200_000,
    validation_window: int = 50_000,
    test_window: int = 50_000,
    embargo: int = 500,
    step: int | None = None,
) -> FoldWindows:
    """Return walk-forward windows that fit the dataset, scaling down if needed.

    When the requested windows are larger than the dataset (a common case for
    short lookbacks), shrink them proportionally so the pipeline still runs
    instead of raising.
    """
    required = train_window + validation_window + test_window + 2 * embargo
    if required <= total:
        return FoldWindows(train_window, validation_window, test_window, embargo, step)
    available = max(total - 2 * embargo, 3)
    factor = available / (train_window + validation_window + test_window)
    tw = max(100, int(train_window * factor))
    vw = max(50, int(validation_window * factor))
    sw = max(50, int(test_window * factor))
    return FoldWindows(
        train_window=tw,
        validation_window=vw,
        test_window=sw,
        embargo=min(embargo, max(0, (total - tw - vw - sw) // 2)),
        step=step if step is None else min(step, max(sw, 1)),
    )


# ---------------------------------------------------------------------------
# Behavioural cloning: distill the profitable persona into the policy.
#
# Raw RL kept failing (in-sample +30%, out-of-sample -62%) because a small MLP
# cannot rediscover a multi-feature market edge from scratch in 20k steps. The
# persona already holds the edge. Behavioural cloning pretrains the policy to
# COPY the persona's decisions (supervised), after which PPO only needs to
# fine-tune — the standard "imitation -> RL" recipe. No hardcoded trade cap is
# needed: the policy inherits the persona's selective, dynamic behaviour.
# ---------------------------------------------------------------------------


def persona_actions_for_bars(bars: pd.DataFrame) -> list[int]:
    """Map the persona-adaptive champion's ACTUAL entries to env actions.

    Runs the managed backtest (the champion's own engine, with its cooldown +
    side state + gates) and maps each entry fill to its bar. 0 = hold, 1 = enter
    long, 2 = enter short. This is the faithful champion reproduction — a naive
    direct ``on_bar`` loop would never reset the persona's side after a managed
    exit and would under-generate entries by ~10x.
    """
    from slytrade.backtest.engine import BacktestConfig
    from slytrade.backtest.reporting import run_managed_aligned_backtest_from_bars
    from slytrade.tasks import _persona_config_from_risk, _trade_config_from_risk

    symbol = str(bars["symbol"].iloc[0]) if "symbol" in bars.columns else "XAUUSD"
    timeframe = str(bars["timeframe"].iloc[0]) if "timeframe" in bars.columns else "H1"
    # Synthetic / non-aligned bars (no decision_time) can't run the backtest:
    # fall back to the direct strategy loop (fine for tests; the faithful
    # backtest path is used on the pipeline's aligned bars).
    if "decision_time" not in bars.columns:
        from slytrade.strategies.personality_adaptive import PersonalityAdaptiveStrategy

        strategy = PersonalityAdaptiveStrategy(symbol=symbol, volume=0.1, config=_persona_config_from_risk(symbol, timeframe))
        actions: list[int] = []
        for i in range(len(bars)):
            intent = strategy.on_bar(i, bars.iloc[i])
            actions.append(0 if intent is None else (1 if intent.side.value == "buy" else 2))
        return actions
    result = run_managed_aligned_backtest_from_bars(
        bars,
        strategy_name="persona-adaptive",
        symbol=symbol,
        volume=0.1,
        point_value=100.0,
        config=BacktestConfig(
            initial_balance=100_000.0,
            point_size=0.01,
            point_value=100.0,
            commission_per_volume=3.5,
            slippage_points=5,
        ),
        trade_config=_trade_config_from_risk(timeframe),
        persona_config=_persona_config_from_risk(symbol, timeframe),
    )
    # Map each entry fill's event time to the bar that emitted it (bar-open
    # grid, side="right"-1). Robust UTC conversion: the ledger event times are
    # tz-aware UTC; a naive one is assumed UTC.
    from datetime import UTC

    bar_opens = pd.to_datetime(bars["time"], utc=True).to_numpy(dtype="datetime64[ns]")
    actions = [0] * len(bars)
    for record in result.trades:
        if not record.reason.startswith("persona_"):
            continue
        dt = record.event_time
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        event_ns = np.datetime64(dt.replace(tzinfo=None), "ns")
        index = int(np.searchsorted(bar_opens, event_ns, side="right") - 1)
        if 0 <= index < len(actions):
            actions[index] = 1 if record.side.value == "buy" else 2
    return actions


def collect_demonstrations(env, actions: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Collect (observation, persona action) pairs by stepping the env.

    Observations come from the SAME env used for training, so they are already
    normalised with the fold's scaler — no leakage, no mismatch.
    """
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    obs, _ = env.reset()
    obs_list.append(np.asarray(obs, dtype=np.float32))
    act_list.append(actions[0])
    for i in range(1, min(len(actions), len(env.bars))):
        obs, _, terminated, truncated, _ = env.step(actions[i - 1])
        if terminated or truncated:
            obs, _ = env.reset()
        obs_list.append(np.asarray(obs, dtype=np.float32))
        act_list.append(actions[i])
    return np.stack(obs_list), np.asarray(act_list, dtype=np.int64)


def _clone_step(model, x: Any, y: Any, optimizer: Any, loss_fn: Any, batch: int) -> None:
    """One minibatch of supervised imitation (shared by BC and DAgger)."""
    import torch

    policy = model.policy
    n = len(x)
    perm = torch.randperm(n)
    for start in range(0, n, batch):
        idx = perm[start : start + batch]
        features = policy.extract_features(x[idx])
        latent, _ = policy.mlp_extractor(features)
        logits = policy.action_net(latent)
        loss = loss_fn(logits, y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def behavioral_clone(model, obs: np.ndarray, act: np.ndarray, *, epochs: int = 10, lr: float = 1e-3, batch: int = 256,
                     weight_decay: float = 1e-4) -> None:
    """Supervised pretrain of the policy network to imitate the persona.

    Top-tier imitation-learning practices for a heavy class imbalance
    (~99% hold / ~1% entries):

    * inverse-sqrt-freq class weights (milder than inverse-freq, so the rare
      entry class is upweighted without over-amplifying label noise),
    * class-balanced minibatch upsampling,
    * AdamW with weight decay (regularisation that survives the imbalance),
    * a linear learning-rate decay over the epochs (no overshooting the rare
      classes late in training).
    """
    import torch

    policy = model.policy
    x = torch.as_tensor(obs, dtype=torch.float32)
    y = torch.as_tensor(act, dtype=torch.long)

    # 4 actions (hold/long/short/flatten); the persona only emits {0,1,2} but
    # the policy's action_net outputs 4 logits, so the weight tensor must match.
    counts = torch.bincount(y, minlength=4).float()
    weights = torch.sqrt(counts.sum() / (counts + 1e-6)).clamp(max=30.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    # Upsample each class so every minibatch is roughly class-balanced.
    idx_by_class = [torch.nonzero(y == c, as_tuple=False).flatten() for c in range(4)]
    largest = max(len(idx) for idx in idx_by_class)
    balanced: list[torch.Tensor] = []
    for idx in idx_by_class:
        if len(idx) == 0:
            continue
        repeats = max(1, int(round(largest / len(idx))))
        balanced.append(idx.repeat(repeats))
    flat = torch.cat(balanced)
    perm = torch.randperm(len(flat))
    flat = flat[perm]
    xb, yb = x[flat], y[flat]

    for epoch in range(epochs):
        # Linear LR decay: halve the step size over the full schedule.
        for g in optimizer.param_groups:
            g["lr"] = lr * (1.0 - epoch / max(epochs, 1))
        _clone_step(model, xb, yb, optimizer, loss_fn, batch)


def dagger_refine(
    model,
    env: SlyTradeRLEnvironment,
    actions: list[int],
    *,
    epochs: int = 5,
    lr: float = 5e-4,
    batch: int = 256,
    weight_decay: float = 1e-4,
) -> None:
    """One DAgger-style refinement pass (fixes behavioural-cloning drift).

    Rolls the current policy out through the env to see WHERE it actually goes
    (its own state distribution), labels those visited states with the persona's
    action, and retrains on that mixture. This attacks the compounding-error
    problem — the reason a clone trained only on the persona's exact states
    diverges the moment it makes one slightly-off decision at inference.
    """
    import torch

    policy = model.policy
    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    obs, _ = env.reset()
    visited = 0
    for i in range(min(len(actions), len(env.bars))):
        obs_list.append(np.asarray(obs, dtype=np.float32))
        act_list.append(actions[i])  # the persona's label at this bar
        action, _ = model.predict(obs, deterministic=False)
        obs, _, terminated, truncated, _ = env.step(int(action))
        visited += 1
        if terminated or truncated:
            obs, _ = env.reset()
    if visited == 0:
        return

    x = torch.as_tensor(np.stack(obs_list), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(act_list, dtype=np.int64), dtype=torch.long)
    counts = torch.bincount(y, minlength=4).float()
    weights = torch.sqrt(counts.sum() / (counts + 1e-6)).clamp(max=30.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        for g in optimizer.param_groups:
            g["lr"] = lr * (1.0 - epoch / max(epochs, 1))
        _clone_step(model, x, y, optimizer, loss_fn, batch)


# ---------------------------------------------------------------------------
# Training (stable-baselines3, lazy import)
# ---------------------------------------------------------------------------

def _as_vec_env(env: SlyTradeRLEnvironment, n_envs: int):
    """Wrap an environment in parallel subprocess vector envs.

    Each worker gets its own copy of the environment (rollout diversity), and
    SB3 steps them in parallel so PPO's data collection is n_envs times faster.
    ``env`` itself is left untouched (the single-env evaluation paths use it).

    ``fork`` is requested explicitly (the SB3 default ``forkserver`` is
    fragile in some environments); if spawning workers fails for any reason,
    training degrades gracefully to single-process rollouts instead of
    crashing a long run.
    """
    import copy
    import logging

    from stable_baselines3.common.vec_env import SubprocVecEnv

    logger = logging.getLogger("slytrade.rl")
    try:
        return SubprocVecEnv([lambda: copy.deepcopy(env) for _ in range(n_envs)], start_method="fork")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("parallel rollouts unavailable (%s); falling back to single-process", exc)
        return env


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
    progress_bar: bool = False,
    n_envs: int = 1,
    warmstart_persona: bool = False,
    bc_epochs: int = 10,
    dagger_passes: int = 0,
    fine_tune_lr: float | None = None,
    fine_tune_ent_coef: float = 0.02,
):
    """Train a PPO policy on the environment. Returns the trained model.

    ``policy_type`` selects the network: "mlp" (default, core SB3) or "lstm"
    (recurrent, regime memory). LSTM policies are NOT part of core
    stable-baselines3 — they live in ``sb3-contrib`` (RecurrentPPO), so the
    ``rl`` extra must include ``sb3-contrib``.

    ``n_envs > 1`` collects rollouts from parallel subprocess copies of the
    environment, which speeds up PPO data collection several-fold on
    multi-core machines (total_timesteps are spread across the workers).
    """
    normalized = policy_type.strip().lower()
    training_env = _as_vec_env(env, n_envs) if n_envs > 1 else env
    model: Any
    if normalized in ("lstm", "recurrent", "mlplstm"):
        try:
            from sb3_contrib import RecurrentPPO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "LSTM policies require sb3-contrib. Install it with:  pip install 'slytrade-rl-bot[rl]'"
            ) from exc
        model = RecurrentPPO(
            "MlpLstmPolicy",
            training_env,
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
    else:
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            training_env,
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

    if warmstart_persona and normalized != "lstm":
        # Superbrain distillation: behavioural-clone the profitable persona
        # (top-tier BC: inverse-sqrt class weights, class-balanced minibatches,
        # AdamW + weight decay, LR decay), then DAgger-refine to kill
        # distribution shift, then a GENTLE RL fine-tune (small LR + entropy
        # bonus) that can improve without unlearning the edge.
        actions = persona_actions_for_bars(env.bars)
        observations, demonstrations = collect_demonstrations(env, actions)
        behavioral_clone(model, observations, demonstrations, epochs=bc_epochs)
        for _ in range(max(0, dagger_passes)):
            dagger_refine(model, env, actions)

    if warmstart_persona and normalized != "lstm" and fine_tune_lr is not None:
        # Gentler fine-tune on top of the distilled policy: a low LR keeps the
        # policy close to the profitable clone; a positive entropy coefficient
        # keeps exploration from collapsing to a single deterministic action.
        model.learning_rate = fine_tune_lr
        if hasattr(model, "ent_coef"):
            model.ent_coef = fine_tune_ent_coef
        model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
    else:
        model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
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
    progress_bar: bool = False,
):
    """Train a policy with any supported algorithm (PPO/SAC/TD3).

    All stable-baselines3 imports are lazy so the core package keeps working
    without the `rl` extras. Algorithm-specific hyperparameters are passed via
    ``policy_kwargs`` (the caller decides what makes sense for the algorithm).
    """
    if policy_type.strip().lower() in ("lstm", "recurrent", "mlplstm"):
        raise ValueError("LSTM/recurrent policies are only supported for PPO; use train_ppo with policy_type='lstm'")
    cls = _sb3_class(algorithm)
    model = cls("MlpPolicy", env, seed=seed, policy_kwargs=policy_kwargs or {}, verbose=verbose)
    model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
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
        recurrent_state = None
        episode_start = True
        while not done and not truncated:
            action, recurrent_state = model.predict(
                obs, state=recurrent_state, episode_start=episode_start, deterministic=True
            )
            episode_start = False
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
    progress: bool = False,
    progress_bar: bool = False,
    dynamic_features: bool = True,
    correlation_threshold: float = 0.92,
    n_envs: int = 1,
    n_seeds: int = 1,
    shaping_enabled: bool = False,
    max_trades_per_episode: int = 0,
    warmstart_persona: bool = False,
) -> pd.DataFrame:
    """Train PPO on each fold's training window, evaluate on its test window.

    The scaler AND the feature selection are fitted on each fold's train slice
    only (no leakage). With ``dynamic_features`` the observation is restricted
    to the footprint-significant columns for that fold; the count emerges from
    the data (threshold-free shadow significance), never from a hardcoded size.

    ``n_seeds > 1`` trains that many independent policies per fold (different
    seeds) and reports the AVERAGE out-of-sample result. A single PPO run on a
    noisy reward has huge seed variance — the champion-vs-RL decision should
    never hinge on one random draw.
    """
    from slytrade.rl.environment import RLEnvironmentConfig

    rows: list[dict] = []
    # The RL must play the SAME exit game as the champion it is measured
    # against: adopt the validated per-timeframe profile's stop/target/hold,
    # otherwise it optimises a different (worse) structure than the persona
    # backtest shown in rl_minus_persona.
    from slytrade.config.timeframe_profiles import profile_for
    from slytrade.rl.dataset import infer_timeframe

    profile = profile_for(infer_timeframe(dataset.bars))
    env_config = RLEnvironmentConfig(
        seed=seed,
        reward_type=reward_type,
        shaping_enabled=shaping_enabled,
        max_trades_per_episode=max_trades_per_episode,
        stop_loss_atr=profile.stop_loss_atr,
        take_profit_atr=profile.take_profit_atr,
        max_bars_in_trade=profile.max_bars_in_trade or 0,
        round_trip_cost_r=profile.cost_per_trade_r,
        entry_cooldown_bars=profile.cooldown_bars,
    )

    for index, fold in enumerate(folds):
        if progress:
            from slytrade.progress import progress as _progress

            _progress(index + 1, len(folds), f"fold {fold.index}: train [{fold.train_start}, {fold.train_end}) → test [{fold.test_start}, {fold.test_end})")
        # Fit the scaler on this fold's train window ONLY.
        scaler_params = dataset.fit_scaler(fold.train_start, fold.train_end)
        selected: list[str] | None = None
        if dynamic_features:
            selected = list(dataset.select_features_on_fold(fold.train_start, fold.train_end, correlation_threshold=correlation_threshold))
            if progress:
                from slytrade.progress import info as _info

                _info(f"  selected {len(selected)}/{len(dataset.features.columns)} features (footprint significance)")

        seed_returns: list[float] = []
        seed_trades: list[float] = []
        seed_drawdowns: list[float] = []
        for seed_index in range(max(1, n_seeds)):
            run_seed = seed + fold.index + seed_index * 1000
            train_env = dataset.env_factory(
                fold.train_start,
                fold.train_end,
                seed=run_seed,
                scaler_params=scaler_params,
                config=env_config,
                feature_columns=selected,
            )
            model = train_ppo(
                train_env,
                total_timesteps=total_timesteps,
                seed=run_seed,
                policy_type=policy_type,
                progress_bar=progress_bar,
                n_envs=n_envs,
                warmstart_persona=warmstart_persona,
            )

            test_env = dataset.env_factory(
                fold.test_start,
                fold.test_end,
                seed=run_seed,
                scaler_params=scaler_params,
                config=env_config,
                feature_columns=selected,
            )
            results = evaluate_ppo(model, test_env, episodes=3, seed=run_seed)
            seed_returns.append(float(results["mean_total_return"]))
            seed_trades.append(float(results["mean_n_trades"]))
            seed_drawdowns.append(float(results["mean_max_drawdown"]))

        row = {
            "fold": fold.index,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "test_mean_total_return": float(np.mean(seed_returns)),
            "test_mean_n_trades": float(np.mean(seed_trades)),
            "test_mean_max_drawdown": float(np.mean(seed_drawdowns)),
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
