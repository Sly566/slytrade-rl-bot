from datetime import UTC, datetime

import pandas as pd

from slytrade.data.storage import MarketDataStorage


def test_tick_storage_path(tmp_path):
    storage = MarketDataStorage(tmp_path)
    path = storage.tick_path("XAUUSD", datetime(2026, 1, 2, tzinfo=UTC))

    assert path.as_posix().endswith("mt5_ticks/symbol=XAUUSD/year=2026/month=01/day=02.parquet")


def test_bar_storage_path(tmp_path):
    storage = MarketDataStorage(tmp_path)
    path = storage.bar_path("XAUUSD", "M1", datetime(2026, 1, 2, tzinfo=UTC))

    assert path.as_posix().endswith("mt5_bars/symbol=XAUUSD/timeframe=M1/year=2026/month=01/day=02.parquet")


def test_write_frame_fallback_or_parquet(tmp_path):
    storage = MarketDataStorage(tmp_path)
    frame = pd.DataFrame({"a": [1, 2]})
    result = storage.write_frame(frame, tmp_path / "sample.parquet")

    assert result.rows == 2
    assert result.path.exists()
    assert result.format in {"parquet", "csv"}
