import pandas as pd

from slytrade.backtest.engine import BacktestConfig
from slytrade.backtest.tick_engine import TickBacktestEngine
from slytrade.execution.models import OrderIntent, Side


def test_tick_engine_matches_exness_base_symbol_to_mt5_suffix():
    bars = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-08-09 22:00:00", "2026-08-09 22:01:00"], utc=True),
            "symbol": ["XAUUSDm", "XAUUSDm"],
            "timeframe": ["M1", "M1"],
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],
        }
    )
    ticks = pd.DataFrame(
        {
            "time_msc": pd.to_datetime(
                ["2026-08-09 22:00:59.000", "2026-08-09 22:01:59.000"], utc=True
            ),
            "symbol": ["XAUUSD", "XAUUSD"],
            "bid": [99.9, 100.1],
            "ask": [100.1, 100.3],
        }
    )

    class BuyOnce:
        def on_bar(self, index: int, bar: pd.Series):
            if index == 0:
                return OrderIntent(symbol="XAUUSDm", side=Side.BUY, volume=0.1, reason="test")
            return None

    result = TickBacktestEngine(
        BacktestConfig(max_quote_age_seconds=5.0, allow_bar_quote_fallback=False)
    ).run(bars, ticks, BuyOnce())
    assert len(result.reports) == 1
    assert result.reports[0].message != "no fresh quote available"
