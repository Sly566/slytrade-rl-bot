from dataclasses import dataclass
from datetime import UTC, datetime

from slytrade.data.mt5_collectors import MT5BarCollector, MT5TickCollector
from slytrade.data.storage import MarketDataStorage


@dataclass(frozen=True)
class FakeSymbol:
    name: str
    description: str = ""


class FakeMT5:
    COPY_TICKS_ALL = 1
    TIMEFRAME_M1 = 101

    def symbols_get(self):
        return [FakeSymbol("XAUUSDm", "Gold")]

    def symbol_select(self, symbol, enabled):
        return True

    def copy_ticks_range(self, symbol, start, end, flag):
        assert symbol == "XAUUSDm"
        assert flag == self.COPY_TICKS_ALL
        return [
            {"time": int(start.timestamp()), "time_msc": int(start.timestamp() * 1000), "bid": 2400.0, "ask": 2400.2, "last": 0.0, "volume": 1},
        ]

    def copy_rates_range(self, symbol, timeframe, start, end):
        assert symbol == "XAUUSDm"
        assert timeframe == self.TIMEFRAME_M1
        return [
            {"time": int(start.timestamp()), "open": 2400.0, "high": 2401.0, "low": 2399.0, "close": 2400.5, "tick_volume": 10},
        ]


def test_tick_collector_writes_file(tmp_path):
    collector = MT5TickCollector(FakeMT5(), MarketDataStorage(tmp_path))
    result = collector.collect("XAUUSD", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC))

    assert result.rows == 1
    assert result.file_count == 1
    assert result.files[0].path.exists()


def test_bar_collector_writes_file(tmp_path):
    collector = MT5BarCollector(FakeMT5(), MarketDataStorage(tmp_path))
    result = collector.collect("XAUUSD", "M1", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC), chunk_size="day")

    assert result.rows == 1
    assert result.file_count == 1
    assert result.files[0].path.exists()


def test_unsupported_timeframe_raises(tmp_path):
    collector = MT5BarCollector(FakeMT5(), MarketDataStorage(tmp_path))
    try:
        collector.timeframe_constant("BAD")
    except ValueError as exc:
        assert "Unsupported timeframe" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
