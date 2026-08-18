"""Adapter symbol resolution must pick the standard contract and never hard-crash
on symbol_select returning False (the root cause of the live-loop startup crash).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from slytrade.brokers.mt5_adapter import MT5BrokerAdapter
from slytrade.execution.oms import OrderManagementSystem
from slytrade.risk.guardrails import GuardrailConfig, TradingGuardrails


class _SymbolInfo(SimpleNamespace):
    pass


def _symbol(name: str) -> _SymbolInfo:
    return _SymbolInfo(
        custom=False, chart_mode=0, select=False, visible=False, bid=0.0, ask=0.0,
        point=0.001, trade_tick_value=1.6, trade_contract_size=100.0,
        volume_min=0.01, volume_max=200.0, volume_step=0.01, name=name,
        path=f"Metals\\{name}", description=f"desc {name}",
    )


class FakeMT5:
    """Returns raw SymbolInfo objects (like the real bridge) and symbol_select=False."""

    def __init__(self) -> None:
        self.symbols = [_symbol("XAUUSDm"), _symbol("XAUUSD247m"), _symbol("EURUSDm"), _symbol("XAUAUDm")]

    def initialize(self) -> bool:
        return True

    def symbols_get(self):
        return self.symbols

    def symbol_select(self, name: str, enable: bool = True) -> bool:
        # Mirrors the failing bridge: Market Watch can't add it.
        return False

    def symbol_info(self, name: str):
        return _symbol(name)


def _adapter() -> MT5BrokerAdapter:
    guardrails = TradingGuardrails(
        GuardrailConfig(allow_live_trading=True),
        initial_equity=1000.0,
        kill_switch_path="/tmp/ks.json",
    )
    return MT5BrokerAdapter(FakeMT5(), oms=OrderManagementSystem(), guardrails=guardrails, allow_trading=True)


def test_resolve_symbol_prefers_standard_contract() -> None:
    resolved = _adapter().resolve_symbol("XAUUSD")
    assert resolved == "XAUUSDm"  # not XAUUSD247m, not XAUAUDm


def test_resolve_symbol_does_not_crash_on_symbol_select_false() -> None:
    adapter = _adapter()
    resolved = adapter.resolve_symbol("XAUUSD")
    assert resolved == "XAUUSDm"
    # health must record a non-fatal note (healthy=True), not raise
    assert adapter.health.statuses["mt5"].healthy is True


def test_resolve_symbol_exact_and_other_symbols() -> None:
    adapter = _adapter()
    assert adapter.resolve_symbol("EURUSD") == "EURUSDm"
    assert adapter.resolve_symbol("XAUAUD") == "XAUAUDm"


def test_resolve_symbol_unknown_raises() -> None:
    adapter = _adapter()
    with pytest.raises(RuntimeError, match="no MT5 symbol matches"):
        adapter.resolve_symbol("BTCUSD")
