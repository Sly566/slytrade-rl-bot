"""Tests for the breakeven-at-R exit rule."""
from __future__ import annotations

import pandas as pd

from slytrade.backtest.trade_management import (
    ManagedTradeState,
    TradeManagementConfig,
    update_breakeven_stop,
)
from slytrade.execution.models import Side


def make_long() -> ManagedTradeState:
    return ManagedTradeState(
        symbol="XAUUSD",
        side=Side.BUY,
        initial_volume=1.0,
        remaining_volume=1.0,
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        partial_take_profit=None,
        entry_index=0,
    )


def make_short() -> ManagedTradeState:
    return ManagedTradeState(
        symbol="XAUUSD",
        side=Side.SELL,
        initial_volume=1.0,
        remaining_volume=1.0,
        entry_price=100.0,
        stop_loss=101.0,
        take_profit=98.0,
        partial_take_profit=None,
        entry_index=0,
    )


def _bar(high: float, low: float) -> pd.Series:
    return pd.Series({"high": high, "low": low, "atr": 1.0})


def test_breakeven_moves_long_stop_to_entry_after_1r():
    trade = make_long()
    update_breakeven_stop(trade, _bar(101.5, 99.5), TradeManagementConfig(breakeven_at_r=1.0))
    assert trade.breakeven_applied
    assert trade.stop_loss == 100.0


def test_breakeven_does_not_trigger_before_1r():
    trade = make_long()
    update_breakeven_stop(trade, _bar(100.5, 99.5), TradeManagementConfig(breakeven_at_r=1.0))
    assert not trade.breakeven_applied
    assert trade.stop_loss == 99.0


def test_breakeven_moves_short_stop_to_entry_after_1r():
    trade = make_short()
    update_breakeven_stop(trade, _bar(99.5, 98.5), TradeManagementConfig(breakeven_at_r=1.0))
    assert trade.breakeven_applied
    assert trade.stop_loss == 100.0


def test_breakeven_never_lowers_long_stop():
    # The stop must never move DOWN for a long (no widening) — it only ratchets
    # toward entry.
    trade = make_long()
    update_breakeven_stop(trade, _bar(101.5, 99.5), TradeManagementConfig(breakeven_at_r=1.0))
    update_breakeven_stop(trade, _bar(99.0, 98.0), TradeManagementConfig(breakeven_at_r=1.0))
    assert trade.stop_loss == 100.0


def test_breakeven_default_off():
    assert TradeManagementConfig().breakeven_at_r is None
    trade = make_long()
    update_breakeven_stop(trade, _bar(101.5, 99.5), TradeManagementConfig())
    assert not trade.breakeven_applied
    assert trade.stop_loss == 99.0


def test_breakeven_rejects_bad_config():
    import pytest

    with pytest.raises(ValueError):
        TradeManagementConfig(breakeven_at_r=0.0)
