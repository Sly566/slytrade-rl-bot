from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from slytrade.rl.deployment import (
    ModelArtifact,
    ModelArtifactMeta,
    promote_artifact,
    save_model_artifact,
    sha256_file,
)
from slytrade.rl.governance import ModelRegistry


class FakeModel:
    def save(self, path: str) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", b"fake-model-bytes")


def test_artifact_meta_roundtrip() -> None:
    meta = ModelArtifactMeta(
        model_id="ppo-XAUUSD-42",
        algorithm="ppo",
        symbol="XAUUSD",
        created_at="2026-08-14T00:00:00+00:00",
        feature_columns=("ml_ret_1", "ml_rsi_14"),
        scaler_params={"ml_ret_1": [0.0, 1.0], "ml_rsi_14": [0.5, 0.2]},
        env_config={"reward_type": "risk_adjusted"},
        metrics={"mean_total_return": 0.05},
        artifact_hash="abc",
        training_data_hash="def",
    )
    restored = ModelArtifactMeta.from_json(meta.to_json())
    assert restored.model_id == meta.model_id
    assert restored.scaler_params == {"ml_ret_1": (0.0, 1.0), "ml_rsi_14": (0.5, 0.2)}
    assert restored.feature_columns == ("ml_ret_1", "ml_rsi_14")


def test_sha256_file(tmp_path: Path) -> None:
    file = tmp_path / "m.txt"
    file.write_text("hello", encoding="utf-8")
    digest = sha256_file(file)
    assert len(digest) == 64
    file.write_text("changed", encoding="utf-8")
    assert sha256_file(file) != digest


def test_save_and_register_artifact(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    registry = tmp_path / "registry.jsonl"
    record = save_model_artifact(
        FakeModel(),
        model_id="ppo-XAUUSD-42",
        algorithm="ppo",
        symbol="XAUUSD",
        feature_columns=["ml_ret_1", "ml_rsi_14"],
        scaler_params={"ml_ret_1": (0.0, 1.0), "ml_rsi_14": (0.5, 0.2)},
        env_config={"reward_type": "risk_adjusted"},
        metrics={"mean_total_return": 0.05},
        artifacts_dir=artifacts,
        registry_path=registry,
        training_data_hash="data-hash",
    )
    assert record["model_id"] == "ppo-XAUUSD-42"
    manifest = artifacts / "ppo-XAUUSD-42" / "manifest.json"
    assert manifest.exists()
    meta = ModelArtifactMeta.from_json(manifest.read_text(encoding="utf-8"))
    assert meta.artifact_hash == record["artifact_hash"]
    # Registry verifies its own hash chain.
    assert ModelRegistry(registry).verify()


def test_duplicate_model_rejected(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    registry = tmp_path / "registry.jsonl"
    save_model_artifact(FakeModel(), model_id="m1", algorithm="ppo", symbol="XAUUSD", feature_columns=["a"], scaler_params={"a": (0.0, 1.0)}, env_config={}, metrics={}, artifacts_dir=artifacts, registry_path=registry)
    with pytest.raises(ValueError, match="already registered"):
        save_model_artifact(FakeModel(), model_id="m1", algorithm="ppo", symbol="XAUUSD", feature_columns=["a"], scaler_params={"a": (0.0, 1.0)}, env_config={}, metrics={}, artifacts_dir=artifacts, registry_path=registry)


def test_promote_requires_evidence(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    registry = tmp_path / "registry.jsonl"
    save_model_artifact(FakeModel(), model_id="m1", algorithm="ppo", symbol="XAUUSD", feature_columns=["a"], scaler_params={"a": (0.0, 1.0)}, env_config={}, metrics={}, artifacts_dir=artifacts, registry_path=registry)
    # Promotion without evidence is refused.
    with pytest.raises(ValueError, match="promotion requires approved lockbox"):
        promote_artifact("m1", registry_path=registry, lockbox_passed=False)
    record = promote_artifact("m1", registry_path=registry, stage="paper")
    assert record["stage"] == "paper"


def test_load_model_artifact_requires_sb3(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    artifacts = tmp_path / "artifacts"
    registry = tmp_path / "registry.jsonl"
    save_model_artifact(FakeModel(), model_id="m1", algorithm="ppo", symbol="XAUUSD", feature_columns=["a"], scaler_params={"a": (0.0, 1.0)}, env_config={}, metrics={}, artifacts_dir=artifacts, registry_path=registry)
    artifact = ModelArtifact(artifacts / "m1", "m1")
    assert artifact.feature_columns == ("a",)
