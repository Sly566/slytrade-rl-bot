from types import SimpleNamespace

from slytrade.brokers.mt5_adapter import MT5BrokerAdapter
from slytrade.execution.models import OrderIntent, OrderStatus, Side
from slytrade.execution.oms import OrderManagementSystem
from slytrade.risk.guardrails import GuardrailConfig, TradingGuardrails


class FakeMT5:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010

    def initialize(self):
        return True

    def account_info(self):
        return SimpleNamespace(equity=100_000)

    def symbol_info(self, symbol):
        return SimpleNamespace(
            name=symbol, digits=2, point=0.01, trade_tick_size=0.01,
            trade_tick_value=1.0, trade_contract_size=100.0,
            volume_min=0.01, volume_max=10.0, volume_step=0.01,
        )

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=2400.0, ask=2400.2, time=1770000000)

    def positions_get(self):
        return ()

    def orders_get(self):
        return ()

    def order_send(self, request):
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, volume=request["volume"], price=request["price"], order=7, comment="done")


def test_mt5_adapter_requires_reconciliation_and_is_idempotent():
    mt5 = FakeMT5()
    oms = OrderManagementSystem()
    adapter = MT5BrokerAdapter(
        mt5,
        oms=oms,
        guardrails=TradingGuardrails(GuardrailConfig(allow_live_trading=True), 100_000),
        allow_trading=True,
    )
    adapter.connect()
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.1, client_order_id="mt5-idempotent")
    quote = adapter.quote("XAUUSD")

    blocked = adapter.submit(intent, quote)
    assert blocked.status == OrderStatus.REJECTED
    assert "reconciliation" in blocked.message

    assert adapter.reconcile().reconciled
    filled = adapter.submit(OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.1, client_order_id="mt5-filled"), quote)
    assert filled.status == OrderStatus.FILLED
    again = adapter.submit(OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.1, client_order_id="mt5-filled"), quote)
    assert again.message == "idempotent existing order"
