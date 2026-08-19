"""Dashboard settings store — the operator's live configuration.

The dashboard holds the bot's tunable configuration in a small JSON file
(``state/dashboard_settings.json``) instead of baking defaults into the code:

* ``symbols``            — the watchlist the bot trades (comma list in the UI)
* ``timeframe``          — the decision timeframe (M15 by default, validated)
* ``risk_per_trade``     — fraction of equity risked per trade (0.005 = 0.5%)
* ``max_position_volume``— per-order volume cap (lots)
* ``limit_entry_atr``    — the champion's limit-entry pullback (0.25 ATR, validated)
* ``lookback``           — history window for pipeline stages (1y)
* ``loop_command``       — what the supervised loop runs (paper | live)

Every field is validated on load and on save; bad values are rejected with a
clear message rather than silently accepted. The file is written atomically.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DASHBOARD_SETTINGS_PATH = "state/dashboard_settings.json"
VALID_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1")
VALID_LOOP_COMMANDS = ("paper", "live", "paper-multi", "live-multi")


@dataclass
class DashboardSettings:
    symbols: list[str] = field(default_factory=lambda: ["XAUUSD"])
    timeframe: str = "M15"
    risk_per_trade: float = 0.005
    max_position_volume: float = 1.0
    limit_entry_atr: float = 0.25
    lookback: str = "1y"
    loop_command: str = "live"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DashboardSettings:
        symbols = data.get("symbols") or ["XAUUSD"]
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        return cls(
            symbols=symbols or ["XAUUSD"],
            timeframe=str(data.get("timeframe") or "M15").upper(),
            risk_per_trade=float(data.get("risk_per_trade", 0.005)),
            max_position_volume=float(data.get("max_position_volume", 1.0)),
            limit_entry_atr=float(data.get("limit_entry_atr", 0.25)),
            lookback=str(data.get("lookback") or "1y"),
            loop_command=str(data.get("loop_command") or "live").lower(),
        )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.symbols:
            problems.append("symbols: at least one symbol is required")
        if any(len(s) < 2 or not s.replace("_", "").replace("-", "").isalnum() for s in self.symbols):
            problems.append(f"symbols: invalid symbol name in {self.symbols}")
        if self.timeframe not in VALID_TIMEFRAMES:
            problems.append(f"timeframe: must be one of {list(VALID_TIMEFRAMES)}, got {self.timeframe!r}")
        if not (0 < self.risk_per_trade <= 0.1):
            problems.append(f"risk_per_trade: must be in (0, 0.1], got {self.risk_per_trade}")
        if not (0 < self.max_position_volume <= 500):
            problems.append(f"max_position_volume: must be in (0, 500], got {self.max_position_volume}")
        if not (0 <= self.limit_entry_atr <= 5):
            problems.append(f"limit_entry_atr: must be in [0, 5], got {self.limit_entry_atr}")
        if self.loop_command not in VALID_LOOP_COMMANDS:
            problems.append(f"loop_command: must be one of {list(VALID_LOOP_COMMANDS)}, got {self.loop_command!r}")
        return problems


def default_dashboard_settings(env: dict[str, str] | None = None) -> DashboardSettings:
    """Defaults from the environment (and configs/risk.yaml when present)."""
    env = env or dict(os.environ)
    syms_env = str(env.get("SLYTRADE_SYMBOLS") or env.get("SLYTRADE_SYMBOL") or "XAUUSD")
    symbols = [s.strip().upper() for s in syms_env.split(",") if s.strip()] or ["XAUUSD"]
    timeframe = str(env.get("SLYTRADE_TIMEFRAME") or "M15").upper()
    risk = 0.005
    max_vol = 1.0
    limit_atr = 0.25
    loop = "live" if env.get("SLYTRADE_DASHBOARD_COMMAND", "").lower() == "live" else "paper"
    try:
        from slytrade.core.config import load_config

        risk_cfg = load_config("configs").risk
        entry = (risk_cfg.get("ict", {}) or {}).get("entry", {}) or {}
        risk = float(risk_cfg.get("risk_per_trade", risk))
        max_vol = float(risk_cfg.get("max_position_volume", max_vol))
        limit_atr = float(entry.get("limit_entry_atr", limit_atr) or limit_atr)
    except Exception:  # pragma: no cover - config dir optional in tests
        pass
    return DashboardSettings(
        symbols=symbols,
        timeframe=timeframe,
        risk_per_trade=risk,
        max_position_volume=max_vol,
        limit_entry_atr=limit_atr,
        lookback="1y",
        loop_command=loop,
    )


def load_dashboard_settings(path: str | Path, env: dict[str, str] | None = None) -> DashboardSettings:
    """Load settings from the file, falling back to defaults for missing fields."""
    settings = default_dashboard_settings(env)
    p = Path(path)
    if not p.exists():
        return settings
    try:
        raw = json.loads(p.read_text())
    except Exception:  # pragma: no cover - corrupt file
        return settings
    if not isinstance(raw, dict):
        return settings
    merged = settings.to_dict()
    merged.update({k: v for k, v in raw.items() if v is not None})
    candidate = DashboardSettings.from_dict(merged)
    problems = candidate.validate()
    # Corrupt/invalid stored values fall back to safe defaults field-by-field.
    if not problems:
        return candidate
    return settings


def save_dashboard_settings(path: str | Path, data: dict[str, Any], env: dict[str, str] | None = None) -> tuple[DashboardSettings, list[str]]:
    """Validate and atomically persist settings. Returns (settings, problems)."""
    base = load_dashboard_settings(path, env)
    merged = base.to_dict()
    merged.update({k: v for k, v in data.items() if k in merged})
    candidate = DashboardSettings.from_dict(merged)
    problems = candidate.validate()
    if problems:
        return base, problems
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(candidate.to_dict(), indent=2))
    os.replace(tmp, p)
    return candidate, []
