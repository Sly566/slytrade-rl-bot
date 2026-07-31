from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SymbolSpec:
    """Broker symbol trading specification used for realistic PnL and sizing.

    For MT5 symbols, `point_value_per_price_unit` is normally:

        trade_tick_value / trade_tick_size

    This is the monetary PnL value for a 1.0 price move at volume 1.0.
    """

    name: str
    digits: int
    point: float
    trade_tick_size: float
    trade_tick_value: float
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    currency_base: str = ""
    currency_profit: str = ""
    currency_margin: str = ""
    description: str = ""
    source: str = "mt5"
    captured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def point_value_per_price_unit(self) -> float:
        if self.trade_tick_size <= 0:
            raise ValueError("trade_tick_size must be positive")
        return self.trade_tick_value / self.trade_tick_size

    def normalize_volume(self, volume: float) -> float:
        """Clamp and round volume to broker min/max/step."""
        if volume <= 0:
            raise ValueError("volume must be positive")
        step = self.volume_step if self.volume_step > 0 else 0.01
        clamped = min(max(volume, self.volume_min), self.volume_max)
        steps = round((clamped - self.volume_min) / step)
        normalized = self.volume_min + steps * step
        return round(min(max(normalized, self.volume_min), self.volume_max), 10)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True)
class BacktestPricing:
    point_size: float
    point_value: float
    volume_min: float
    volume_max: float
    volume_step: float


def spec_to_backtest_pricing(spec: SymbolSpec) -> BacktestPricing:
    return BacktestPricing(
        point_size=spec.trade_tick_size if spec.trade_tick_size > 0 else spec.point,
        point_value=spec.point_value_per_price_unit,
        volume_min=spec.volume_min,
        volume_max=spec.volume_max,
        volume_step=spec.volume_step,
    )


def symbol_spec_from_mt5_info(info: Any, *, source: str = "mt5") -> SymbolSpec:
    if info is None:
        raise ValueError("symbol_info returned None")
    return SymbolSpec(
        name=str(info.name),
        digits=int(getattr(info, "digits", 0)),
        point=float(getattr(info, "point", 0.0)),
        trade_tick_size=float(getattr(info, "trade_tick_size", 0.0)),
        trade_tick_value=float(getattr(info, "trade_tick_value", 0.0)),
        trade_contract_size=float(getattr(info, "trade_contract_size", 0.0)),
        volume_min=float(getattr(info, "volume_min", 0.0)),
        volume_max=float(getattr(info, "volume_max", 0.0)),
        volume_step=float(getattr(info, "volume_step", 0.0)),
        currency_base=str(getattr(info, "currency_base", "")),
        currency_profit=str(getattr(info, "currency_profit", "")),
        currency_margin=str(getattr(info, "currency_margin", "")),
        description=str(getattr(info, "description", "")),
        source=source,
    )


def save_symbol_spec(spec: SymbolSpec, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(spec.to_json(), encoding="utf-8")
    return output


def load_symbol_spec(path: str | Path) -> SymbolSpec:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SymbolSpec(**data)
