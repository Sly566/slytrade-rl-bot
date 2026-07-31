from dataclasses import dataclass

from slytrade.brokers.specs import (
    SymbolSpec,
    load_symbol_spec,
    save_symbol_spec,
    spec_to_backtest_pricing,
    symbol_spec_from_mt5_info,
)


@dataclass(frozen=True)
class FakeInfo:
    name: str = "XAUUSDm"
    digits: int = 3
    point: float = 0.001
    trade_tick_size: float = 0.001
    trade_tick_value: float = 1.6532609999999999
    trade_contract_size: float = 100.0
    volume_min: float = 0.01
    volume_max: float = 200.0
    volume_step: float = 0.01
    currency_base: str = "XAU"
    currency_profit: str = "USD"
    currency_margin: str = "XAU"
    description: str = "Gold vs US Dollar"


def test_symbol_spec_from_mt5_info_and_point_value():
    spec = symbol_spec_from_mt5_info(FakeInfo())

    assert spec.name == "XAUUSDm"
    assert spec.point_value_per_price_unit == 1653.2609999999997
    pricing = spec_to_backtest_pricing(spec)
    assert pricing.point_size == 0.001
    assert pricing.point_value == spec.point_value_per_price_unit


def test_symbol_spec_normalize_volume():
    spec = SymbolSpec(
        name="XAUUSDm",
        digits=3,
        point=0.001,
        trade_tick_size=0.001,
        trade_tick_value=1.0,
        trade_contract_size=100,
        volume_min=0.01,
        volume_max=1.0,
        volume_step=0.01,
    )

    assert spec.normalize_volume(0.001) == 0.01
    assert spec.normalize_volume(0.026) == 0.03
    assert spec.normalize_volume(2.0) == 1.0


def test_save_and_load_symbol_spec(tmp_path):
    spec = symbol_spec_from_mt5_info(FakeInfo())
    path = save_symbol_spec(spec, tmp_path / "XAUUSDm.json")
    loaded = load_symbol_spec(path)

    assert loaded == spec
