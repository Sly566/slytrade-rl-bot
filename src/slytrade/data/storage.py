from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

DatasetKind = Literal["ticks", "bars"]


@dataclass(frozen=True)
class WriteResult:
    path: Path
    rows: int
    format: str


class MarketDataStorage:
    """Partitioned on-disk storage for raw tick and bar data."""

    def __init__(self, root: str | Path = "data/raw"):
        self.root = Path(root)

    def tick_path(self, symbol: str, start: datetime, extension: str = "parquet") -> Path:
        return (
            self.root
            / "mt5_ticks"
            / f"symbol={symbol}"
            / f"year={start.year:04d}"
            / f"month={start.month:02d}"
            / f"day={start.day:02d}.{extension}"
        )

    def bar_path(self, symbol: str, timeframe: str, start: datetime, extension: str = "parquet") -> Path:
        return (
            self.root
            / "mt5_bars"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"year={start.year:04d}"
            / f"month={start.month:02d}"
            / f"day={start.day:02d}.{extension}"
        )

    def write_frame(self, df: pd.DataFrame, preferred_path: Path) -> WriteResult:
        preferred_path.parent.mkdir(parents=True, exist_ok=True)
        if preferred_path.suffix == ".parquet":
            try:
                df.to_parquet(preferred_path, index=False)
                return WriteResult(preferred_path, len(df), "parquet")
            except Exception:
                fallback = preferred_path.with_suffix(".csv")
                df.to_csv(fallback, index=False)
                return WriteResult(fallback, len(df), "csv")
        df.to_csv(preferred_path, index=False)
        return WriteResult(preferred_path, len(df), "csv")

    def write_ticks(self, symbol: str, start: datetime, df: pd.DataFrame) -> WriteResult:
        return self.write_frame(df, self.tick_path(symbol, start))

    def write_bars(self, symbol: str, timeframe: str, start: datetime, df: pd.DataFrame) -> WriteResult:
        return self.write_frame(df, self.bar_path(symbol, timeframe, start))
