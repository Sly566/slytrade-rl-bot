"""Trader Personality Configuration Loader"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TraderPersonality:
    name: str = "SlyMasterICT"
    description: str = "Adaptive scalping and day trading ICT specialist"

    aggression: float = 0.65
    selectivity: float = 0.75
    risk_tolerance: float = 0.60

    scalping_bias: float = 0.70
    day_trading_bias: float = 0.30

    macro_respect: float = 0.85
    session_sensitivity: float = 0.80

    confluence_style: str = "balanced"

    adaptive_rules: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str = "configs/trader_personality.yaml") -> "TraderPersonality":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        personality_data = data.get("trader_personality", {})
        return cls(**personality_data)
