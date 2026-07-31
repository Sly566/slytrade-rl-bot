import pandas as pd
from typer.testing import CliRunner

from slytrade.backtest.engine import BacktestConfig, BuyAndHoldOnceStrategy
from slytrade.backtest.trade_management import (
    ManagedAlignedBacktestEngine,
    TradeManagementConfig,
    create_trade_state,
    exit_reason_for_bar,
)
from slytrade.cli import app
from slytrade.execution.models import OrderIntent, Side


def make_aligned_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-01", periods=5, freq="min", tz="UTC"),
            "decision_time": pd.date_range("2026-07-01T00:01:00Z", periods=5, freq="min"),
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "atr": [1.0, 1.0, 1.0, 1.0, 1.0],
            "quote_time": pd.date_range("2026-07-01T00:01:00Z", periods=5, freq="min"),
            "quote_bid": [100.0, 101.0, 102.0, 103.0, 104.0],
            "quote_ask": [100.2, 101.2, 102.2, 103.2, 104.2],
            "quote_mid": [100.1, 101.1, 102.1, 103.1, 104.1],
            "quote_spread": [0.2] * 5,
            "quote_age_seconds": [0.0] * 5,
            "quote_is_fresh": [True] * 5,
            "tick_mid_high": [100.5, 103.5, 104.0, 105.0, 106.0],
            "tick_mid_low": [99.8, 100.8, 101.8, 102.8, 103.8],
        }
    )


def test_exit_reason_take_profit_for_long():
    bars = make_aligned_bars()
    trade = create_trade_state(
        OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=1.0),
        100.0,
        bars.iloc[0],
        0,
        TradeManagementConfig(stop_loss_atr=1.0, take_profit_atr=2.0),
    )

    assert exit_reason_for_bar(trade, bars.iloc[1], 1, TradeManagementConfig()) == "take_profit"


def test_exit_reason_stop_loss_for_long():
    bars = make_aligned_bars()
    trade = create_trade_state(
        OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=1.0),
        102.0,
        bars.iloc[0],
        0,
        TradeManagementConfig(stop_loss_atr=1.0, take_profit_atr=2.0),
    )

    assert exit_reason_for_bar(trade, bars.iloc[1], 1, TradeManagementConfig()) == "stop_loss"


def test_managed_backtest_creates_entry_and_exit():
    engine = ManagedAlignedBacktestEngine(
        BacktestConfig(initial_balance=100_000, point_value=1.0),
        TradeManagementConfig(stop_loss_atr=1.0, take_profit_atr=2.0),
    )
    result = engine.run(make_aligned_bars(), BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=1.0))

    assert result.metrics.trades >= 2
    assert result.orders[0].intent.reason == "buy_and_hold_once"
    assert any(order.intent.reason.startswith("managed_") for order in result.orders)


def test_managed_backtest_cli(tmp_path):
    bars_path = tmp_path / "aligned.csv"
    make_aligned_bars().to_csv(bars_path, index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "run-managed-backtest",
            "--bars-file",
            str(bars_path),
            "--strategy",
            "buy-and-hold",
            "--stop-loss-atr",
            "1.0",
            "--take-profit-atr",
            "2.0",
        ],
    )

    assert result.exit_code == 0
    assert "Backtest Report" in result.stdout
