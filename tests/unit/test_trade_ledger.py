from datetime import UTC, datetime

from slytrade.execution.journal import JsonlJournal
from slytrade.execution.ledger import TradeLedger
from slytrade.execution.models import OrderIntent, Side


def test_trade_ledger_records_fill():
    ledger = TradeLedger()
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.1, reason="test")

    record = ledger.record_fill(
        intent,
        volume=0.1,
        price=2400.0,
        commission=1.0,
        realized_pnl=0.0,
        event_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert record.client_order_id == intent.client_order_id
    assert ledger.total_commission == 1.0
    assert ledger.total_realized_pnl == 0.0
    assert len(ledger.to_frame()) == 1


def test_trade_ledger_journal(tmp_path):
    journal = JsonlJournal(tmp_path / "trades.jsonl")
    ledger = TradeLedger(journal)
    intent = OrderIntent(symbol="XAUUSD", side=Side.SELL, volume=0.2)

    ledger.record_fill(
        intent,
        volume=0.2,
        price=2399.0,
        commission=2.0,
        realized_pnl=5.0,
        event_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    rows = journal.read_all()
    assert rows[0]["event_type"] == "trade_record"
    assert rows[0]["trade"]["realized_pnl"] == 5.0
