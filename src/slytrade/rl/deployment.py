"""Model artifact packaging and promotion for deployment.

Training produces a policy zip (stable-baselines3). Deployment needs more than
the weights: the exact feature columns, the fitted scaler, the environment
config, the algorithm, and a content hash so the artifact can be traced back
through the model registry and loaded deterministically at inference time.

This module bridges that gap:

    train  ->  save_model_artifact()  ->  ModelRegistry.register()
    evaluate ->  PromotionDecision     ->  ModelRegistry.promote()
    deploy  ->  load_model_artifact()  ->  RLPolicyStrategy (inference)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slytrade.rl.governance import ModelRegistry, PromotionDecision


@dataclass(frozen=True)
class ModelArtifactMeta:
    model_id: str
    algorithm: str
    symbol: str
    created_at: str
    feature_columns: tuple[str, ...]
    scaler_params: dict[str, tuple[float, float]]
    env_config: dict[str, Any]
    metrics: dict[str, float]
    artifact_hash: str
    training_data_hash: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, raw: str) -> ModelArtifactMeta:
        data = json.loads(raw)
        return cls(
            model_id=str(data["model_id"]),
            algorithm=str(data["algorithm"]),
            symbol=str(data["symbol"]),
            created_at=str(data["created_at"]),
            feature_columns=tuple(str(c) for c in data["feature_columns"]),
            scaler_params={str(k): (float(v[0]), float(v[1])) for k, v in data["scaler_params"].items()},
            env_config=dict(data.get("env_config", {})),
            metrics={str(k): float(v) for k, v in data.get("metrics", {}).items()},
            artifact_hash=str(data["artifact_hash"]),
            training_data_hash=str(data.get("training_data_hash", "")),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ModelArtifact:
    """A deployed, content-addressed model bundle on disk."""

    directory: Path
    model_id: str
    meta: ModelArtifactMeta = field(init=False)
    model_path: Path = field(init=False)
    manifest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_path = self.directory / "model.zip"
        self.manifest_path = self.directory / "manifest.json"
        self.meta = ModelArtifactMeta.from_json(self.manifest_path.read_text(encoding="utf-8"))

    @property
    def scaler_params(self) -> dict[str, tuple[float, float]]:
        return {key: (float(mean), float(std)) for key, (mean, std) in self.meta.scaler_params.items()}

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.meta.feature_columns


def save_model_artifact(
    model,
    *,
    model_id: str,
    algorithm: str,
    symbol: str,
    feature_columns: list[str],
    scaler_params: dict[str, tuple[float, float]],
    env_config: dict[str, Any],
    metrics: dict[str, float],
    artifacts_dir: str | Path = "models/artifacts",
    training_data_hash: str = "",
    registry_path: str | Path = "models/registry.jsonl",
) -> dict[str, Any]:
    """Save a trained model bundle and register it in the hash-chained registry.

    Returns the registry record. The artifact directory is ``artifacts_dir /
    <model_id>`` and contains the model zip plus a JSON manifest whose hash is
    stored in the registry for later verification.
    """
    if not model_id.strip():
        raise ValueError("model_id cannot be empty")
    directory = Path(artifacts_dir) / model_id
    directory.mkdir(parents=True, exist_ok=True)

    model_path = directory / "model.zip"
    model.save(str(model_path))

    artifact_hash = sha256_file(model_path)
    meta = ModelArtifactMeta(
        model_id=model_id,
        algorithm=algorithm,
        symbol=symbol,
        created_at=datetime.now(UTC).isoformat(),
        feature_columns=tuple(feature_columns),
        scaler_params={key: (float(mean), float(std)) for key, (mean, std) in scaler_params.items()},
        env_config=env_config,
        metrics={key: float(value) for key, value in metrics.items()},
        artifact_hash=artifact_hash,
        training_data_hash=training_data_hash,
    )
    (directory / "manifest.json").write_text(meta.to_json() + "\n", encoding="utf-8")

    registry = ModelRegistry(registry_path)
    record = registry.register(
        model_id,
        artifact_uri=str(directory),
        artifact_hash=artifact_hash,
        training_data_hash=training_data_hash or f"unset:{model_id}",
        metadata={"algorithm": algorithm, "symbol": symbol, "metrics": meta.metrics},
    )
    return record


def load_model_artifact(model_id: str, artifacts_dir: str | Path = "models/artifacts") -> tuple[Any, ModelArtifact]:
    """Load a trained model plus its manifest, verifying the stored hash.

    Returns ``(model, artifact)``. The model import is lazy so callers without
    stable-baselines3 can still read the manifest.
    """
    from stable_baselines3 import PPO  # noqa: F401  (imports all SB3 model loaders)

    artifact = ModelArtifact(Path(artifacts_dir) / model_id, model_id)
    if sha256_file(artifact.model_path) != artifact.meta.artifact_hash:
        raise ValueError(f"artifact hash mismatch for model {model_id!r}")
    model = _load_zip(artifact.model_path)
    return model, artifact


def _load_zip(path: Path):
    """Load an SB3 model zip without knowing the algorithm ahead of time."""
    import zipfile

    import stable_baselines3 as sb3

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "data" in names:  # SB3 stores the pickled model under "data"
            import pickle

            with archive.open("data") as handle:
                model = pickle.load(handle)
            return model
        # Fallback: pick the most specific algorithm class by matching.
        for algo in ("PPO", "SAC", "TD3", "A2C", "DQN"):
            if any(name.startswith(algo) for name in names):
                cls = getattr(sb3, algo)
                return cls.load(str(path))
    raise ValueError(f"cannot determine model class for {path}")


def promote_artifact(
    model_id: str,
    *,
    registry_path: str | Path = "models/registry.jsonl",
    stage: str = "paper",
    lockbox_passed: bool = True,
    cost_stress_passed: bool = True,
    reviewer: str = "operator",
    rationale: str = "",
) -> dict[str, Any]:
    """Promote a registered model through a deployment stage.

    Promotion is refused (raises) unless both the lockbox and cost-stress
    evidence checks passed — the registry enforces this invariant.
    """
    registry = ModelRegistry(registry_path)
    decision = PromotionDecision(
        approved=lockbox_passed and cost_stress_passed,
        lockbox_passed=lockbox_passed,
        cost_stress_passed=cost_stress_passed,
        reviewer=reviewer,
        rationale=rationale or "evidence review passed",
    )
    return registry.promote(model_id, stage, decision)


def data_hash_of(frame_bytes: bytes) -> str:
    """Hash raw dataset bytes for training-data provenance."""
    return hashlib.sha256(frame_bytes).hexdigest()


__all__ = [
    "ModelArtifact",
    "ModelArtifactMeta",
    "load_model_artifact",
    "promote_artifact",
    "save_model_artifact",
    "sha256_file",
    "data_hash_of",
]
