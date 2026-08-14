"""Optional MLflow experiment tracking for RL training.

MLflow is declared in the ``rl`` extras but must never be a hard dependency:
training runs on machines without MLflow still work, they just do not record a
tracking run. Every helper is a safe no-op when MLflow is absent.
"""

from __future__ import annotations

import importlib.util
from typing import Any


def mlflow_available() -> bool:
    return importlib.util.find_spec("mlflow") is not None


def maybe_start_run(experiment_name: str, run_name: str | None = None) -> Any | None:
    """Start an MLflow run if available; otherwise return None."""
    if not mlflow_available():
        return None
    import mlflow

    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def maybe_log_params(run: Any | None, params: dict[str, Any]) -> None:
    if run is None or not mlflow_available():
        return
    import mlflow

    mlflow.log_params({str(key): value for key, value in params.items()})


def maybe_log_metrics(run: Any | None, metrics: dict[str, float], step: int | None = None) -> None:
    if run is None or not mlflow_available():
        return
    import mlflow

    mlflow.log_metrics({str(key): float(value) for key, value in metrics.items()}, step=step)


def maybe_end_run(run: Any | None) -> None:
    if run is None or not mlflow_available():
        return
    import mlflow

    mlflow.end_run()
