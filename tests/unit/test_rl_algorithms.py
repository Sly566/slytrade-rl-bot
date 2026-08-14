from __future__ import annotations

import pytest

from slytrade.rl.walkforward import SUPPORTED_ALGORITHMS, resolve_algorithm, train_policy


def test_supported_algorithms() -> None:
    assert set(SUPPORTED_ALGORITHMS) == {"ppo", "sac", "td3"}


def test_resolve_algorithm_normalises_case_and_whitespace() -> None:
    assert resolve_algorithm("PPO") == "ppo"
    assert resolve_algorithm(" Sac ") == "sac"
    assert resolve_algorithm("TD3") == "td3"


def test_resolve_algorithm_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported algorithm"):
        resolve_algorithm("dqn")


def test_train_policy_rejects_unknown_without_imports() -> None:
    # Unknown algorithms fail fast (before any stable-baselines3 import).
    with pytest.raises(ValueError):
        train_policy("dqn", object())


def test_policy_class_resolution() -> None:
    from slytrade.rl.walkforward import _policy_class

    assert _policy_class("mlp") == "MlpPolicy"
    assert _policy_class("lstm") == "MlpLstmPolicy"
    assert _policy_class("recurrent") == "MlpLstmPolicy"
    import pytest

    with pytest.raises(ValueError):
        _policy_class("transformer")
