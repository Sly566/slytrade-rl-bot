"""Evaluation and model-governance primitives for the RL stack.

The module is deliberately independent of SB3 and the trading environment.  A
caller supplies a small evaluator function, which makes the controls usable in
CI as well as for expensive offline runs.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CostScenario:
    """Trading-cost assumptions, expressed in basis points per one-way trade."""

    name: str
    spread_bps: float = 0.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    @property
    def total_bps(self) -> float:
        return self.spread_bps + self.commission_bps + self.slippage_bps

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("cost scenario name cannot be empty")
        if min(self.spread_bps, self.commission_bps, self.slippage_bps) < 0:
            raise ValueError("cost assumptions cannot be negative")


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class EvaluationReport:
    seeds: tuple[int, ...]
    metrics: Mapping[str, MetricSummary]
    scenario: str | None = None
    split: str = "test"

    def passed(self, metric: str, minimum: float | None = None, maximum: float | None = None) -> bool:
        summary = self.metrics[metric]
        return (minimum is None or summary.ci_low >= minimum) and (
            maximum is None or summary.ci_high <= maximum
        )


def _z_value(confidence: float) -> float:
    # Avoid a scipy dependency in the core package.  These values are the
    # standard normal quantiles used for the usual governance confidence levels.
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    return {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(round(confidence, 2), 1.96)


def aggregate_seed_metrics(
    results: Sequence[Mapping[str, float]], *, confidence: float = 0.95
) -> dict[str, MetricSummary]:
    """Aggregate scalar metrics across independent seeds with confidence intervals."""
    if not results:
        raise ValueError("at least one seed result is required")
    keys = set(results[0])
    if any(set(result) != keys for result in results):
        raise ValueError("all seed results must contain the same metrics")
    z = _z_value(confidence)
    output: dict[str, MetricSummary] = {}
    for key in sorted(keys):
        values = np.asarray([float(result[key]) for result in results], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"metric {key!r} contains a non-finite value")
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half_width = z * std / math.sqrt(len(values))
        output[key] = MetricSummary(len(values), mean, std, mean - half_width, mean + half_width)
    return output


def _call_evaluator(evaluator: Callable[..., Mapping[str, float]], seed: int, scenario: CostScenario | None, split: str):
    """Call evaluators with either the simple ``(seed)`` or governed signature."""
    names = set(inspect.signature(evaluator).parameters)
    kwargs: dict[str, Any] = {}
    if "seed" in names:
        kwargs["seed"] = seed
    if "scenario" in names:
        kwargs["scenario"] = scenario
    if "split" in names:
        kwargs["split"] = split
    if kwargs:
        return evaluator(**kwargs)
    return evaluator(seed)


def evaluate_seeds(
    evaluator: Callable[..., Mapping[str, float]],
    seeds: Sequence[int],
    *,
    confidence: float = 0.95,
    scenario: CostScenario | None = None,
    split: str = "test",
) -> EvaluationReport:
    """Evaluate a policy for every seed and aggregate all scalar metrics."""
    unique_seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not unique_seeds:
        raise ValueError("at least one seed is required")
    results = [_call_evaluator(evaluator, seed, scenario, split) for seed in unique_seeds]
    return EvaluationReport(unique_seeds, aggregate_seed_metrics(results, confidence=confidence), scenario.name if scenario else None, split)


@dataclass(frozen=True)
class LockboxSpec:
    """An immutable, content-addressed evaluation holdout."""

    name: str
    data_hash: str
    sample_count: int

    def __post_init__(self) -> None:
        if not self.name or not self.data_hash or self.sample_count <= 0:
            raise ValueError("lockbox requires a name, data hash, and positive sample count")


def evaluate_lockbox(
    evaluator: Callable[..., Mapping[str, float]],
    seeds: Sequence[int],
    lockbox: LockboxSpec,
    *,
    confidence: float = 0.95,
    training_data_hash: str | None = None,
) -> EvaluationReport:
    """Run a holdout evaluation and reject accidental train/lockbox overlap."""
    if training_data_hash and training_data_hash == lockbox.data_hash:
        raise ValueError("lockbox data hash must differ from training data hash")
    return evaluate_seeds(evaluator, seeds, confidence=confidence, split=f"lockbox:{lockbox.name}")


def evaluate_cost_stress(
    evaluator: Callable[..., Mapping[str, float]],
    seeds: Sequence[int],
    scenarios: Sequence[CostScenario],
    *,
    confidence: float = 0.95,
) -> dict[str, EvaluationReport]:
    """Evaluate identical seeds under each cost assumption."""
    if not scenarios:
        raise ValueError("at least one cost scenario is required")
    if len({scenario.name for scenario in scenarios}) != len(scenarios):
        raise ValueError("cost scenario names must be unique")
    return {
        scenario.name: evaluate_seeds(evaluator, seeds, confidence=confidence, scenario=scenario)
        for scenario in scenarios
    }


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    lockbox_passed: bool
    cost_stress_passed: bool
    reviewer: str
    rationale: str


class ModelRegistry:
    """Append-only, hash-chained JSONL registry.

    Existing records are never edited or removed.  Every append includes the
    previous record hash, allowing a deployment job to verify the audit trail.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    def _records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def verify(self) -> bool:
        previous = ""
        for record in self._records():
            payload = dict(record)
            digest = payload.pop("record_hash")
            if payload["previous_hash"] != previous:
                return False
            if hashlib.sha256(_canonical(payload).encode()).hexdigest() != digest:
                return False
            previous = digest
        return True

    def _append(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        records = self._records()
        previous = records[-1]["record_hash"] if records else ""
        body = {"event": event, "timestamp": datetime.now(UTC).isoformat(), **payload, "previous_hash": previous}
        record = {**body, "record_hash": hashlib.sha256(_canonical(body).encode()).hexdigest()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def register(self, model_id: str, *, artifact_uri: str, artifact_hash: str, training_data_hash: str, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not model_id or not artifact_hash or not training_data_hash:
            raise ValueError("model_id and content hashes are required")
        if any(r.get("event") == "register" and r.get("model_id") == model_id for r in self._records()):
            raise ValueError(f"model {model_id!r} is already registered")
        return self._append("register", {"model_id": model_id, "artifact_uri": artifact_uri, "artifact_hash": artifact_hash, "training_data_hash": training_data_hash, "metadata": dict(metadata or {})})

    def promote(self, model_id: str, stage: str, decision: PromotionDecision) -> dict[str, Any]:
        if not decision.approved or not decision.lockbox_passed or not decision.cost_stress_passed:
            raise ValueError("promotion requires approved lockbox and cost-stress checks")
        if not any(r.get("event") == "register" and r.get("model_id") == model_id for r in self._records()):
            raise ValueError(f"unknown model {model_id!r}")
        return self._append("promote", {"model_id": model_id, "stage": stage, "decision": asdict(decision)})


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
