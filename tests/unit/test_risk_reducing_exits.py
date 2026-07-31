from slytrade.backtest.execution import ExecutionConfig, Quote
from slytrade.execution.models import OrderIntent, OrderStatus, Side
from slytrade.execution.paper_broker import PaperBroker
from slytrade.risk.guardrails import GuardrailConfig


def test_risk_reducing_exit_allowed_after_kill_switch():
    broker = PaperBroker(
        initial_balance=100_000,
        execution_config=ExecutionConfig(point_size=0.01, point_value=1.0),
        guardrail_config=GuardrailConfig(max_total_drawdown=0.0001, max_spread_points=10_000),
    )
    entry = OrderIntent("XAUUSD", Side.BUY, 1.0)
    broker.submit_order(entry, Quote("XAUUSD", bid=100.0, ask=100.2))

    # Force an equity drawdown large enough to arm the kill switch on the next
    # risk-increasing order.
    broker.update_quote(Quote("XAUUSD", bid=80.0, ask=80.2))
    rejected_entry = broker.submit_order(OrderIntent("XAUUSD", Side.BUY, 1.0), Quote("XAUUSD", bid=80.0, ask=80.2))
    assert rejected_entry.report.status == OrderStatus.REJECTED
    assert broker.guardrails.kill_switch

    # The reducing sell must still be allowed, otherwise open risk can get stuck.
    exit_result = broker.submit_order(OrderIntent("XAUUSD", Side.SELL, 1.0), Quote("XAUUSD", bid=80.0, ask=80.2))

    assert exit_result.report.status == OrderStatus.FILLED
    assert "XAUUSD" not in broker.portfolio.positions
