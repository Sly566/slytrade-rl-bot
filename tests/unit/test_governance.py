import json

import pytest

from slytrade.rl.governance import (
    CostScenario,
    LockboxSpec,
    ModelRegistry,
    PromotionDecision,
    aggregate_seed_metrics,
    evaluate_cost_stress,
    evaluate_lockbox,
)


def test_seed_aggregation_has_reproducible_ci():
    summary = aggregate_seed_metrics(
        [{"return": 1.0}, {"return": 2.0}, {"return": 3.0}], confidence=0.95
    )["return"]
    assert summary.n == 3
    assert summary.mean == 2.0
    assert summary.ci_low < 2.0 < summary.ci_high


def test_lockbox_rejects_training_overlap_and_forwards_split():
    calls = []

    def evaluator(seed, split):
        calls.append((seed, split))
        return {"return": float(seed)}

    lockbox = LockboxSpec("2026-q1", "lockbox-hash", 100)
    report = evaluate_lockbox(evaluator, [1, 2], lockbox, training_data_hash="train-hash")
    assert report.split == "lockbox:2026-q1"
    assert calls == [(1, "lockbox:2026-q1"), (2, "lockbox:2026-q1")]
    with pytest.raises(ValueError, match="must differ"):
        evaluate_lockbox(evaluator, [1], lockbox, training_data_hash="lockbox-hash")


def test_cost_stress_uses_named_scenarios():
    reports = evaluate_cost_stress(
        lambda seed, scenario: {"net_return": 10.0 - scenario.total_bps * seed},
        [1, 2],
        [CostScenario("base"), CostScenario("stress", spread_bps=4, slippage_bps=2)],
    )
    assert reports["base"].metrics["net_return"].mean == 10.0
    assert reports["stress"].metrics["net_return"].mean == 1.0


def test_registry_is_append_only_and_hash_chained(tmp_path):
    path = tmp_path / "registry.jsonl"
    registry = ModelRegistry(path)
    registry.register("m1", artifact_uri="models/m1", artifact_hash="sha-a", training_data_hash="data-a")
    registry.promote(
        "m1",
        "paper",
        PromotionDecision(True, True, True, "reviewer", "all gates passed"),
    )
    assert registry.verify()
    assert len(path.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="already registered"):
        registry.register("m1", artifact_uri="other", artifact_hash="sha-b", training_data_hash="data-b")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["artifact_uri"] = "tampered"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    assert not registry.verify()
