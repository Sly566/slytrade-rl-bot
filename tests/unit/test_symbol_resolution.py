from dataclasses import dataclass

import pytest

from slytrade.brokers.symbols import list_matching_symbols, resolve_symbol


@dataclass(frozen=True)
class FakeSymbol:
    name: str
    description: str = ""


class FakeMT5:
    def __init__(self):
        self.selected: list[tuple[str, bool]] = []

    def symbols_get(self):
        return [
            FakeSymbol("EURUSDm", "Euro vs Dollar"),
            FakeSymbol("XAUUSDm", "Gold"),
            FakeSymbol("BTCUSDm", "Bitcoin"),
            FakeSymbol("XAUUSD", "Gold exact"),
        ]

    def symbol_select(self, symbol: str, enabled: bool):
        self.selected.append((symbol, enabled))
        return True


class BridgeMT5(FakeMT5):
    class Container:
        @staticmethod
        def eval(expression: str):
            assert "symbols_get" in expression
            return [{"name": "XAUUSDm", "description": "Gold"}]

    def symbols_get(self):
        raise RuntimeError("Can't pickle SymbolInfo")

    _container = Container()


def test_resolve_prefers_exact_match():
    mt5 = FakeMT5()
    resolved = resolve_symbol(mt5, "XAUUSD")

    assert resolved.resolved == "XAUUSD"
    assert resolved.exact
    assert mt5.selected == [("XAUUSD", True)]


def test_resolve_base_to_shortest_prefixed_symbol_when_no_exact():
    mt5 = FakeMT5()
    resolved = resolve_symbol(mt5, "BTCUSD")

    assert resolved.resolved == "BTCUSDm"
    assert not resolved.exact


def test_list_matching_symbols():
    mt5 = FakeMT5()
    matches = list_matching_symbols(mt5, "XAU")

    assert [match.resolved for match in matches] == ["XAUUSD", "XAUUSDm"]


def test_resolve_unknown_symbol_raises():
    mt5 = FakeMT5()
    with pytest.raises(ValueError):
        resolve_symbol(mt5, "UNKNOWN")


def test_resolve_uses_bridge_safe_symbol_projection():
    resolved = resolve_symbol(BridgeMT5(), "XAUUSD")
    assert resolved.resolved == "XAUUSDm"
    assert resolved.description == "Gold"
