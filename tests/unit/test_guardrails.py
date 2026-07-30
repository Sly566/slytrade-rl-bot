from slytrade.execution.models import OrderIntent, Side
from slytrade.risk.guardrails import GuardrailConfig, TradingGuardrails


def test_live_trading_blocked_by_default():
    guards = TradingGuardrails(GuardrailConfig(), initial_equity=100_000)
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.01)

    decision = guards.approve_order(intent, equity=100_000, live=True)

    assert not decision.approved
    assert "live trading disabled" in decision.reason


def test_order_approved_in_paper_mode():
    guards = TradingGuardrails(GuardrailConfig(), initial_equity=100_000)
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.01)

    decision = guards.approve_order(intent, equity=100_000, live=False)

    assert decision.approved


def test_spread_guard_blocks_bad_trade():
    guards = TradingGuardrails(GuardrailConfig(max_spread_points=10), initial_equity=100_000)
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=0.01)

    decision = guards.approve_order(intent, equity=100_000, spread_points=20)

    assert not decision.approved
    assert "spread too high" in decision.reason
