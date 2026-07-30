from slytrade.backtest.execution import ExecutionConfig, Quote
from slytrade.execution.journal import JsonlJournal
from slytrade.execution.ledger import TradeLedger
from slytrade.execution.models import OrderIntent, OrderStatus, Side
from slytrade.execution.oms import OrderManagementSystem
from slytrade.execution.paper_broker import PaperBroker
from slytrade.risk.guardrails import GuardrailConfig


def test_paper_broker_routes_order_through_oms_portfolio_and_ledger(tmp_path):
    order_journal = JsonlJournal(tmp_path / "orders.jsonl")
    trade_journal = JsonlJournal(tmp_path / "trades.jsonl")
    oms = OrderManagementSystem(order_journal)
    ledger = TradeLedger(trade_journal)
    broker = PaperBroker(
        initial_balance=100_000,
        execution_config=ExecutionConfig(point_size=0.01, point_value=1.0, commission_per_volume=1.0),
        oms=oms,
        ledger=ledger,
    )
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.5)
    quote = Quote(symbol="XAUUSD", bid=100.0, ask=100.2)

    result = broker.submit_order(intent, quote)

    assert result.report.status == OrderStatus.FILLED
    assert oms.get(intent.client_order_id) is not None
    assert len(ledger.records) == 1
    assert broker.portfolio.positions["XAUUSD"].quantity == 0.5
    assert broker.portfolio.total_commission == 0.5
    assert order_journal.read_all()
    assert trade_journal.read_all()


def test_paper_broker_rejects_order_when_guardrails_fail():
    broker = PaperBroker(
        initial_balance=100_000,
        guardrail_config=GuardrailConfig(max_spread_points=1),
        execution_config=ExecutionConfig(point_size=0.01),
    )
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.1)
    quote = Quote(symbol="XAUUSD", bid=100.0, ask=100.5)

    result = broker.submit_order(intent, quote)

    assert result.report.status == OrderStatus.REJECTED
    assert "spread too high" in result.report.message
    assert len(broker.ledger.records) == 0
