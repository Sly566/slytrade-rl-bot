from datetime import UTC, datetime

from rich.console import Console

from slytrade.backtest.analytics import compute_trade_analytics
from slytrade.backtest.engine import BacktestResult
from slytrade.backtest.metrics import compute_performance_metrics
from slytrade.backtest.portfolio import PortfolioState
from slytrade.backtest.reporting import render_trade_analytics
from slytrade.execution.ledger import TradeRecord
from slytrade.execution.models import ExecutionReport, OrderIntent, OrderStatus, Side
from slytrade.execution.oms import OrderManagementSystem


def test_compute_trade_analytics_win_loss_and_reasons():
    records = [
        TradeRecord("entry-1", "XAUUSD", Side.BUY, 1.0, 100.0, 0.5, 0.0, "entry", datetime(2026, 1, 1, tzinfo=UTC)),
        TradeRecord("exit-1", "XAUUSD", Side.SELL, 1.0, 101.0, 0.5, 10.0, "managed_take_profit", datetime(2026, 1, 1, tzinfo=UTC)),
        TradeRecord("entry-2", "XAUUSD", Side.BUY, 1.0, 100.0, 0.5, 0.0, "entry", datetime(2026, 1, 1, tzinfo=UTC)),
        TradeRecord("exit-2", "XAUUSD", Side.SELL, 1.0, 99.0, 0.5, -5.0, "managed_stop_loss", datetime(2026, 1, 1, tzinfo=UTC)),
    ]
    analytics = compute_trade_analytics(records)

    assert analytics.fills == 4
    assert analytics.entry_fills == 2
    assert analytics.exit_fills == 2
    assert analytics.net_realized_pnl == 5.0
    assert analytics.gross_profit == 10.0
    assert analytics.gross_loss == -5.0
    assert analytics.profit_factor == 2.0
    assert analytics.win_rate == 0.5
    assert analytics.exit_reason_counts == {"take_profit": 1, "stop_loss": 1}


def test_order_status_and_reject_reasons_in_analytics():
    oms = OrderManagementSystem()
    intent = OrderIntent("XAUUSD", Side.BUY, 1.0)
    oms.create_order(intent)
    oms.apply_report(ExecutionReport(intent.client_order_id, OrderStatus.REJECTED, message="spread too high"))
    analytics = compute_trade_analytics([], list(oms.orders.values()))

    assert analytics.order_status_counts == {"rejected": 1}
    assert analytics.order_reject_reasons == {"spread too high": 1}


def test_render_trade_analytics(capsys):
    result = BacktestResult(
        equity_curve=[100_000.0, 100_010.0],
        reports=[],
        metrics=compute_performance_metrics([100_000.0, 100_010.0], trades=0),
        final_portfolio=PortfolioState(100_000.0),
        orders=[],
        trades=[],
    )
    render_trade_analytics(result, console=Console())
    captured = capsys.readouterr()

    assert "Trade Analytics" in captured.out
