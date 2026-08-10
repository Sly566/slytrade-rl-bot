from datetime import UTC, datetime

from slytrade.execution.journal import SqliteJournal
from slytrade.execution.ledger import TradeLedger
from slytrade.execution.models import ExecutionReport, OrderIntent, OrderStatus, Side
from slytrade.execution.oms import OrderManagementSystem
from slytrade.risk.guardrails import GuardrailConfig, TradingGuardrails


def test_sqlite_journal_rehydrates_oms_and_ledger(tmp_path):
    journal = SqliteJournal(tmp_path / "execution.sqlite")
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.1, client_order_id="restart-safe-order")
    oms = OrderManagementSystem(journal)
    oms.create_order(intent)
    oms.apply_report(
        ExecutionReport(
            client_order_id=intent.client_order_id,
            status=OrderStatus.FILLED,
            filled_volume=0.1,
            avg_fill_price=2400.0,
            broker_order_id="mt5-1",
            event_time=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )

    restored = OrderManagementSystem(journal)

    assert restored.get(intent.client_order_id) is not None
    assert restored.get(intent.client_order_id).status == OrderStatus.FILLED
    assert restored.get(intent.client_order_id).broker_order_id == "mt5-1"

    ledger = TradeLedger(journal)
    ledger.record_fill(
        intent,
        volume=0.1,
        price=2400.0,
        commission=1.0,
        realized_pnl=0.0,
        event_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert len(TradeLedger(journal).records) == 1


def test_guardrails_trip_on_daily_drawdown():
    guardrails = TradingGuardrails(
        GuardrailConfig(max_daily_drawdown=0.03, max_total_drawdown=0.08),
        initial_equity=100_000,
    )

    decision = guardrails.observe_equity(96_999, current_date=datetime(2026, 1, 1, tzinfo=UTC).date())

    assert not decision.approved
    assert decision.reason == "max daily drawdown breached"
    assert guardrails.kill_switch
