from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    assets: dict[str, Any]
    broker: dict[str, Any]
    data: dict[str, Any]
    risk: dict[str, Any]
    training: dict[str, Any]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_config(config_dir: str | Path = "configs") -> ProjectConfig:
    config_dir = Path(config_dir)

    return ProjectConfig(
        assets=load_yaml(config_dir / "assets.yaml"),
        broker=load_yaml(config_dir / "broker.yaml"),
        data=load_yaml(config_dir / "data.yaml"),
        risk=load_yaml(config_dir / "risk.yaml"),
        training=load_yaml(config_dir / "training.yaml"),
    )
