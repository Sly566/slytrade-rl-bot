from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slytrade.rl.walkforward import resolve_fold_windows


def test_resolve_fold_windows_passes_through_when_fits() -> None:
    windows = resolve_fold_windows(500_000, train_window=200_000, validation_window=50_000, test_window=50_000, embargo=500)
    assert windows.train_window == 200_000
    assert windows.test_window == 50_000


def test_resolve_fold_windows_scales_down_for_small_dataset() -> None:
    # 30k bars is too small for 200k/50k/50k/500 — must scale down instead of raising.
    windows = resolve_fold_windows(30_280, train_window=200_000, validation_window=50_000, test_window=50_000, embargo=500)
    total = windows.train_window + windows.validation_window + windows.test_window + 2 * windows.embargo
    assert total <= 30_280
    assert windows.train_window < 200_000
    assert windows.train_window >= 100
    assert windows.test_window >= 50


def test_resolve_fold_windows_scaled_windows_still_form_fold() -> None:
    from slytrade.rl.walkforward import make_walk_forward_folds

    total = 10_000
    windows = resolve_fold_windows(total, train_window=200_000, validation_window=50_000, test_window=50_000, embargo=100)
    folds = make_walk_forward_folds(
        total,
        train_window=windows.train_window,
        validation_window=windows.validation_window,
        test_window=windows.test_window,
        embargo=windows.embargo,
        step=windows.step,
    )
    assert len(folds) >= 1


def test_train_ppo_lstm_smoke() -> None:
    """Recurrent PPO builds and trains on a tiny environment (regression for
    'Policy MlpLstmPolicy unknown')."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("sb3_contrib")

    from slytrade.rl.environment import RLEnvironmentConfig, SlyTradeRLEnvironment
    from slytrade.rl.walkforward import train_ppo

    times = pd.date_range("2026-08-14T10:00:00", periods=300, freq="min", tz="UTC")
    close = 100.0 + np.linspace(0.0, 1.0, 300)
    bars = pd.DataFrame({"time": times, "symbol": "XAUUSD", "open": close, "high": close + 0.1, "low": close - 0.1, "close": close})
    features = pd.DataFrame({"f1": np.zeros(300, dtype=float), "f2": np.ones(300, dtype=float)})
    env = SlyTradeRLEnvironment(features=features, bars=bars, config=RLEnvironmentConfig(reward_type="trade_pnl"))
    model = train_ppo(env, total_timesteps=256, seed=1, policy_type="lstm", n_steps=128, batch_size=32, n_epochs=2)
    assert model is not None
    obs, _ = env.reset()
    action, _ = model.predict(obs, episode_start=True, deterministic=True)
    assert action in (0, 1, 2, 3)


def test_train_policy_rejects_lstm() -> None:
    from slytrade.rl.walkforward import train_policy

    with pytest.raises(ValueError, match="LSTM/recurrent policies are only supported for PPO"):
        train_policy("sac", object(), policy_type="lstm")
