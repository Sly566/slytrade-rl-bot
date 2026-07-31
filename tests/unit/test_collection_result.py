from dataclasses import dataclass
from datetime import UTC, datetime

from slytrade.data.mt5_collectors import MT5TickCollector
from slytrade.data.storage import MarketDataStorage


@dataclass(frozen=True)
class FakeSymbol:
    name: str
    description: str = ""


class SparseTickMT5:
    COPY_TICKS_ALL = 1

    def symbols_get(self):
        return [FakeSymbol("XAUUSDm", "Gold")]

    def symbol_select(self, symbol, enabled):
        return True

    def copy_ticks_range(self, symbol, start, end, flag):
        # Return data only for the first requested day.
        if start.date().isoformat() == "2026-01-01":
            return [
                {"time": int(start.timestamp()), "time_msc": int(start.timestamp() * 1000), "bid": 1.0, "ask": 1.2, "last": 0.0},
            ]
        return []


def test_collection_result_counts_attempted_and_empty_chunks(tmp_path):
    collector = MT5TickCollector(SparseTickMT5(), MarketDataStorage(tmp_path))

    result = collector.collect(
        "XAUUSD",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 4, tzinfo=UTC),
        chunk_size="day",
    )

    assert result.chunks_attempted == 3
    assert result.empty_chunks == 2
    assert result.rows == 1
    assert result.file_count == 1
