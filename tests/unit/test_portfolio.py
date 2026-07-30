from slytrade.backtest.portfolio import Fill, PortfolioState
from slytrade.execution.models import Side


def test_long_position_realized_pnl():
    portfolio = PortfolioState(initial_balance=100_000)
    portfolio.apply_fill(Fill(symbol="XAUUSD", side=Side.BUY, volume=1.0, price=100.0, point_value=10.0))
    realized = portfolio.apply_fill(Fill(symbol="XAUUSD", side=Side.SELL, volume=1.0, price=101.0, point_value=10.0))

    assert realized == 10.0
    assert portfolio.realized_pnl == 10.0
    assert portfolio.balance == 100_010.0
    assert "XAUUSD" not in portfolio.positions


def test_unrealized_pnl_mark_to_market():
    portfolio = PortfolioState(initial_balance=100_000)
    portfolio.apply_fill(Fill(symbol="XAUUSD", side=Side.BUY, volume=2.0, price=100.0, point_value=5.0))

    assert portfolio.mark_to_market({"XAUUSD": 101.0}) == 100_010.0


def test_position_reversal():
    portfolio = PortfolioState(initial_balance=100_000)
    portfolio.apply_fill(Fill(symbol="XAUUSD", side=Side.BUY, volume=1.0, price=100.0, point_value=1.0))
    portfolio.apply_fill(Fill(symbol="XAUUSD", side=Side.SELL, volume=2.0, price=99.0, point_value=1.0))

    assert portfolio.realized_pnl == -1.0
    assert portfolio.positions["XAUUSD"].quantity == -1.0
    assert portfolio.positions["XAUUSD"].avg_price == 99.0
