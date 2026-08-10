"""Trader Personality Configuration Loader.

The personality is the trader's "persona": a set of traits that modulate how
the strategy and RL layers behave. All fields have safe defaults so the config
can grow without breaking existing code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TraderPersonality:
    name: str = "SlyMasterICT"
    description: str = "Adaptive scalping and day trading ICT specialist"

    # Core Personality Traits (0.0 = low, 1.0 = high)
    aggression: float = 0.65
    selectivity: float = 0.75
    risk_tolerance: float = 0.60

    # Trading Style Bias
    scalping_bias: float = 0.70
    day_trading_bias: float = 0.30

    # Market Context Sensitivity
    macro_respect: float = 0.85
    session_sensitivity: float = 0.80

    # Confluence Style
    confluence_style: str = "balanced"

    # === Deep ICT Persona Traits ===
    # Conviction: how strongly a confirmed setup is acted upon.
    conviction: float = 0.70
    # Patience: willingness to wait for the perfect liquidity sweep + OB entry.
    patience: float = 0.75
    # Discipline: adherence to rules; resists revenge trading after losses.
    discipline: float = 0.85
    # Adaptability: how quickly thresholds shift with changing market regimes.
    adaptability: float = 0.80
    # Time pressure: prefers faster entries/exits (1.0) vs waiting for confirmations (0.0).
    time_pressure: float = 0.35
    # Structure focus: how strongly the trader respects HTF market structure.
    structure_focus: float = 0.90
    # Liquidity focus: how much the trader hunts liquidity sweeps before entries.
    liquidity_focus: float = 0.88
    # Trade duration bias: 0.0 = scalps only, 1.0 = holds for day swings.
    trade_duration_bias: float = 0.45

    # Entry/exit style: "aggressive", "balanced", "conservative"
    entry_style: str = "balanced"
    exit_style: str = "balanced"

    # Risk behavior
    cut_losses_fast: float = 0.75
    let_winners_run: float = 0.60
    max_risk_per_trade_default: float = 0.005
    position_sizing: str = "risk_based"  # risk_based, fixed, kelly

    # Confidence thresholds that map score -> action (regime-adaptive)
    confidence_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "min_entry_score": 4,
            "add_on_strong_conviction": 0.8,
            "scale_out_threshold": 0.6,
        }
    )

    # Preference windows / conditions (used by the adaptive strategy)
    session_preferences: list[str] = field(default_factory=lambda: ["london", "ny_am", "ny_pm"])
    volatility_preferences: list[str] = field(default_factory=lambda: ["normal", "high"])
    trend_preferences: list[str] = field(default_factory=lambda: ["bull", "bear"])

    # Kill-switch / risk circuit breaker
    kill_switch_trigger_daily_drawdown: float = 0.03
    kill_switch_trigger_total_drawdown: float = 0.08

    # Edge optimism (0..1). 0 = skeptical, requires many confirmations; 1 = trusts the edge.
    edge_optimism: float = 0.55

    # === Adaptive Rules (existing field, extended) ===
    adaptive_rules: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate trait ranges and clamp into [0, 1]."""
        for field_name in [
            "aggression", "selectivity", "risk_tolerance",
            "scalping_bias", "day_trading_bias", "macro_respect", "session_sensitivity",
            "conviction", "patience", "discipline", "adaptability", "time_pressure",
            "structure_focus", "liquidity_focus", "trade_duration_bias",
            "cut_losses_fast", "let_winners_run", "edge_optimism",
        ]:
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")
            setattr(self, field_name, round(max(0.0, min(1.0, value)), 4))

        if not 0.0 < self.max_risk_per_trade_default <= 0.05:
            raise ValueError("max_risk_per_trade_default must be in (0, 0.05]")

    def trait(self, name: str, default: float = 0.5) -> float:
        """Safely read a numeric trait, falling back to default."""
        value = getattr(self, name, default)
        return float(value) if value is not None else default

    def adaptive(self, rule_key: str, base: float, default: float = 1.0) -> float:
        """Apply a named multiplicative adaptive rule if present.

        Example: personality.adaptive("high_volatility", "aggression_multiplier")
        falls back to the global rule map if the key is missing.
        """
        rules = self.adaptive_rules or {}
        rule = rules.get(rule_key, {})
        if isinstance(rule, dict):
            return float(rule.get("aggregate", default))
        return float(rule) if rule else default

    def aggregate_multipliers(self, rule_keys: list[str]) -> float:
        """Combine multiplicative multipliers for a set of adaptive rule keys."""
        result = 1.0
        for key in rule_keys:
            rules = self.adaptive_rules or {}
            rule = rules.get(key, {})
            if isinstance(rule, dict):
                for m in rule.values():
                    if isinstance(m, (int, float)) and m != 0:
                        result *= float(m)
            elif isinstance(rule, (int, float)) and rule != 0:
                result *= float(rule)
        return result

    @classmethod
    def from_yaml(cls, path: str = "configs/trader_personality.yaml") -> TraderPersonality:
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("trader_personality", {}))
