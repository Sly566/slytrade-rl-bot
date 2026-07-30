from slytrade.backtest.execution import ExecutionConfig, Quote, TickExecutionSimulator
from slytrade.execution.models import OrderIntent, OrderKind, OrderStatus, Side


def test_market_buy_fills_at_ask_plus_slippage():
    simulator = TickExecutionSimulator(ExecutionConfig(point_size=0.01, slippage_points=2, commission_per_volume=3))
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=1.0)
    quote = Quote(symbol="XAUUSD", bid=100.0, ask=100.1)

    fill = simulator.execute(intent, quote)

    assert fill.report.status == OrderStatus.FILLED
    assert fill.report.avg_fill_price == 100.11999999999999
    assert fill.commission == 3.0


def test_market_sell_fills_at_bid_minus_slippage():
    simulator = TickExecutionSimulator(ExecutionConfig(point_size=0.01, slippage_points=2))
    intent = OrderIntent(symbol="XAUUSD", side=Side.SELL, volume=1.0)
    quote = Quote(symbol="XAUUSD", bid=100.0, ask=100.1)

    fill = simulator.execute(intent, quote)

    assert fill.report.status == OrderStatus.FILLED
    assert fill.report.avg_fill_price == 99.98


def test_limit_order_rests_when_not_crossed():
    simulator = TickExecutionSimulator()
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=1.0, kind=OrderKind.LIMIT, limit_price=99.0)
    quote = Quote(symbol="XAUUSD", bid=100.0, ask=100.1)

    fill = simulator.execute(intent, quote)

    assert fill.report.status == OrderStatus.ACCEPTED
    assert fill.report.filled_volume == 0.0


def test_crossed_spread_rejected():
    simulator = TickExecutionSimulator()
    intent = OrderIntent(symbol="XAUUSD", side=Side.BUY, volume=1.0)
    quote = Quote(symbol="XAUUSD", bid=100.2, ask=100.1)

    fill = simulator.execute(intent, quote)

    assert fill.report.status == OrderStatus.REJECTED
