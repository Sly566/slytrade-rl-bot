import json

from slytrade.execution.journal import JsonlJournal
from slytrade.execution.models import ExecutionReport, OrderIntent, OrderStatus, Side
from slytrade.execution.oms import OrderManagementSystem


def test_oms_create_order_is_idempotent():
    oms = OrderManagementSystem()
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.1)

    first = oms.create_order(intent)
    second = oms.create_order(intent)

    assert first is second
    assert len(oms.orders) == 1
    assert oms.open_orders()[0].client_order_id == intent.client_order_id


def test_oms_apply_execution_report():
    oms = OrderManagementSystem()
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.1)
    oms.create_order(intent)
    report = ExecutionReport(
        client_order_id=intent.client_order_id,
        status=OrderStatus.FILLED,
        filled_volume=0.1,
        avg_fill_price=2400.5,
        broker_order_id="sim-1",
        message="filled",
    )

    state = oms.apply_report(report)

    assert state.status == OrderStatus.FILLED
    assert state.filled_volume == 0.1
    assert state.avg_fill_price == 2400.5
    assert state.broker_order_id == "sim-1"
    assert not state.is_open


def test_oms_journal_writes_jsonl(tmp_path):
    journal = JsonlJournal(tmp_path / "orders.jsonl")
    oms = OrderManagementSystem(journal)
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.1)
    oms.create_order(intent)

    rows = journal.read_all()

    assert len(rows) == 1
    assert rows[0]["event_type"] == "order_created"
    assert json.loads((tmp_path / "orders.jsonl").read_text().splitlines()[0])["event_type"] == "order_created"
