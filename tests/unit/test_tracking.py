from __future__ import annotations

import importlib.util

import pytest

from slytrade.rl.tracking import (
    maybe_end_run,
    maybe_log_metrics,
    maybe_log_params,
    maybe_start_run,
    mlflow_available,
)


def test_tracking_helpers_are_safe_noops_without_mlflow() -> None:
    # MLflow is an optional extra; the helpers must not blow up without it.
    if importlib.util.find_spec("mlflow") is not None:
        pytest.skip("mlflow installed; testing the no-mlflow path is not applicable")
    assert mlflow_available() is False
    run = maybe_start_run("exp")
    assert run is None
    maybe_log_params(run, {"seed": 42})  # no exception
    maybe_log_metrics(run, {"return": 0.1})  # no exception
    maybe_end_run(run)  # no exception


def test_tracking_helpers_accept_none_run_with_mlflow_flag() -> None:
    # Even if mlflow were present, a None run must short-circuit.
    maybe_log_params(None, {"seed": 42})
    maybe_log_metrics(None, {"return": 0.1})
    maybe_end_run(None)
