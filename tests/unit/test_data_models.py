from datetime import UTC, datetime

from slytrade.data.models import Bar, Tick


def test_tick_spread_and_mid():
    tick = Tick(symbol="XAUUSD", time=datetime.now(UTC), bid=2400.0, ask=2400.2)

    assert round(tick.spread, 2) == 0.2
    assert round(tick.mid, 2) == 2400.1


def test_bar_model():
    bar = Bar(
        symbol="XAUUSD",
        timeframe="M1",
        time=datetime.now(UTC),
        open=2400.0,
        high=2401.0,
        low=2399.5,
        close=2400.5,
    )

    assert bar.high >= bar.low
