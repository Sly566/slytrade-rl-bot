from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from slytrade.brokers.symbols import resolve_symbol
from slytrade.data.schemas import normalize_bar_frame, normalize_tick_frame
from slytrade.data.storage import MarketDataStorage, WriteResult
from slytrade.data.time import ChunkSize, iter_time_chunks
from slytrade.data.validators import ValidationReport, validate_bar_frame, validate_tick_frame

TIMEFRAME_ATTRS = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}


@dataclass(frozen=True)
class CollectionResult:
    symbol: str
    dataset: str
    rows: int = 0
    files: list[WriteResult] = field(default_factory=list)
    reports: list[ValidationReport] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)


class MT5TickCollector:
    def __init__(self, mt5: Any, storage: MarketDataStorage | None = None):
        self.mt5 = mt5
        self.storage = storage or MarketDataStorage()

    def collect(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        chunk_size: ChunkSize = "day",
        resolve: bool = True,
    ) -> CollectionResult:
        rows = 0
        files: list[WriteResult] = []
        reports: list[ValidationReport] = []
        copy_flag = self.mt5.COPY_TICKS_ALL
        actual_symbol = resolve_symbol(self.mt5, symbol).resolved if resolve else symbol

        for chunk_start, chunk_end in iter_time_chunks(start, end, chunk_size):
            raw = self.mt5.copy_ticks_range(actual_symbol, chunk_start, chunk_end, copy_flag)
            normalized = normalize_tick_frame(raw, actual_symbol)
            clean, report = validate_tick_frame(normalized)
            reports.append(report)
            if clean.empty:
                continue
            write_result = self.storage.write_ticks(actual_symbol, chunk_start, clean)
            rows += write_result.rows
            files.append(write_result)

        return CollectionResult(symbol=actual_symbol, dataset="ticks", rows=rows, files=files, reports=reports)


class MT5BarCollector:
    def __init__(self, mt5: Any, storage: MarketDataStorage | None = None):
        self.mt5 = mt5
        self.storage = storage or MarketDataStorage()

    def timeframe_constant(self, timeframe: str) -> Any:
        try:
            attr = TIMEFRAME_ATTRS[timeframe]
        except KeyError as exc:
            raise ValueError(f"Unsupported timeframe: {timeframe}") from exc
        return getattr(self.mt5, attr)

    def collect(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        chunk_size: ChunkSize = "month",
        resolve: bool = True,
    ) -> CollectionResult:
        rows = 0
        files: list[WriteResult] = []
        reports: list[ValidationReport] = []
        tf_const = self.timeframe_constant(timeframe)
        actual_symbol = resolve_symbol(self.mt5, symbol).resolved if resolve else symbol

        for chunk_start, chunk_end in iter_time_chunks(start, end, chunk_size):
            raw = self.mt5.copy_rates_range(actual_symbol, tf_const, chunk_start, chunk_end)
            normalized = normalize_bar_frame(raw, actual_symbol, timeframe)
            clean, report = validate_bar_frame(normalized)
            reports.append(report)
            if clean.empty:
                continue
            write_result = self.storage.write_bars(actual_symbol, timeframe, chunk_start, clean)
            rows += write_result.rows
            files.append(write_result)

        return CollectionResult(symbol=actual_symbol, dataset="bars", rows=rows, files=files, reports=reports)
