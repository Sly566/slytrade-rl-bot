import pandas as pd

from slytrade.backtest.engine import BacktestConfig, BuyAndHoldOnceStrategy
from slytrade.backtest.trade_management import (
    ManagedAlignedBacktestEngine,
    TradeManagementConfig,
    create_trade_state,
    exit_reason_for_bar,
    next_exit_event,
    quote_for_exit_price,
    update_trailing_stop,
)
from slytrade.execution.models import OrderIntent, Side


def make_bar(**overrides):
    base = {
        "time": pd.Timestamp("2026-07-01T00:00:00Z"),
        "decision_time": pd.Timestamp("2026-07-01T00:01:00Z"),
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "atr": 1.0,
        "quote_time": pd.Timestamp("2026-07-01T00:01:00Z"),
        "quote_bid": 100.0,
        "quote_ask": 100.2,
        "quote_mid": 100.1,
        "quote_spread": 0.2,
        "quote_age_seconds": 0.0,
        "quote_is_fresh": True,
        "tick_mid_high": 101.0,
        "tick_mid_low": 99.0,
    }
    base.update(overrides)
    return pd.Series(base)


def make_bars() -> pd.DataFrame:
    bars = []
    for i in range(5):
        bars.append(
            make_bar(
                time=pd.Timestamp("2026-07-01T00:00:00Z") + pd.Timedelta(minutes=i),
                decision_time=pd.Timestamp("2026-07-01T00:01:00Z") + pd.Timedelta(minutes=i),
                quote_time=pd.Timestamp("2026-07-01T00:01:00Z") + pd.Timedelta(minutes=i),
                quote_bid=100.0 + i,
                quote_ask=100.2 + i,
                quote_mid=100.1 + i,
                tick_mid_high=101.0 + i,
                tick_mid_low=99.0 + i,
            )
        )
    return pd.DataFrame(bars)


def test_partial_take_profit_event_and_breakeven_state():
    config = TradeManagementConfig(partial_take_profit_enabled=True, partial_take_profit_atr=1.0, partial_close_fraction=0.5)
    trade = create_trade_state(OrderIntent("XAUUSD", Side.BUY, 1.0), 100.0, make_bar(), 0, config)
    event = next_exit_event(trade, make_bar(tick_mid_high=101.1, tick_mid_low=100.0), 1, config)

    assert event == ("partial_take_profit", 0.5, 101.0)


def test_trailing_stop_only_moves_in_favour():
    config = TradeManagementConfig(trailing_stop_atr=1.0)
    trade = create_trade_state(OrderIntent("XAUUSD", Side.BUY, 1.0), 100.0, make_bar(), 0, config)
    original_stop = trade.stop_loss

    update_trailing_stop(trade, make_bar(tick_mid_high=103.0, tick_mid_low=100.0), config)

    assert trade.stop_loss > original_stop
    assert trade.stop_loss == 102.0


def test_same_bar_stop_and_target_is_conservative():
    config = TradeManagementConfig(stop_loss_atr=1.0, take_profit_atr=1.0, conservative_same_bar_exit=True)
    trade = create_trade_state(OrderIntent("XAUUSD", Side.BUY, 1.0), 100.0, make_bar(), 0, config)

    assert exit_reason_for_bar(trade, make_bar(tick_mid_high=101.5, tick_mid_low=98.5), 1, config) == "stop_loss"


def test_quote_for_exit_price_fills_relevant_side():
    from slytrade.backtest.execution import Quote

    trade = create_trade_state(OrderIntent("XAUUSD", Side.BUY, 1.0), 100.0, make_bar(), 0, TradeManagementConfig())
    quote = quote_for_exit_price(trade, 99.0, Quote("XAUUSD", bid=100.0, ask=100.2))

    assert quote.bid == 99.0
    assert quote.ask == 99.2


def test_managed_engine_partial_then_final_exit():
    bars = make_bars()
    engine = ManagedAlignedBacktestEngine(
        BacktestConfig(initial_balance=100_000, point_value=1.0),
        TradeManagementConfig(
            stop_loss_atr=1.0,
            take_profit_atr=2.0,
            partial_take_profit_enabled=True,
            partial_take_profit_atr=1.0,
            partial_close_fraction=0.5,
        ),
    )
    result = engine.run(bars, BuyAndHoldOnceStrategy("XAUUSD", 1.0))
    reasons = [order.intent.reason for order in result.orders]

    assert "managed_partial_take_profit" in reasons
    assert any(reason in reasons for reason in ["managed_take_profit", "managed_trailing_stop", "managed_stop_loss"])
    assert result.metrics.trades >= 3
