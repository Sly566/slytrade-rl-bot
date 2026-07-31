from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from slytrade.data.schemas import TICK_COLUMNS
from slytrade.data.time import ensure_utc

EXNESS_TICK_BASE_URL = "https://ticks.ex2archive.com/ticks"
UrlOpen = Callable[[str], object]


@dataclass(frozen=True)
class ExnessArchiveFile:
    path: Path
    rows: int
    format: str
    period: str


@dataclass(frozen=True)
class ExnessArchiveResult:
    symbol: str
    rows: int = 0
    files: list[ExnessArchiveFile] = field(default_factory=list)
    months_attempted: int = 0
    empty_months: int = 0
    failed_months: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)


def normalize_exness_symbol(symbol: str) -> str:
    """Normalize a symbol for the Exness public tick archive.

    The archive generally uses base symbols like XAUUSD, not broker suffixes
    like XAUUSDm. Keep this conservative: remove whitespace, uppercase, and
    strip a trailing `m` only for common Exness-style CFD symbols.
    """
    normalized = symbol.strip().upper()
    if normalized.endswith("M") and len(normalized) > 6:
        return normalized[:-1]
    return normalized


def iter_month_starts(start: datetime, end: datetime) -> list[datetime]:
    start = ensure_utc(start)
    end = ensure_utc(end)
    if start >= end:
        raise ValueError("start must be earlier than end")
    current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months: list[datetime] = []
    while current < end:
        months.append(current)
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = current.replace(year=year, month=month)
    return months


def build_exness_month_url(symbol: str, year: int, month: int, *, base_url: str = EXNESS_TICK_BASE_URL) -> str:
    archive_symbol = normalize_exness_symbol(symbol)
    month_text = f"{month:02d}"
    return f"{base_url.rstrip('/')}/{archive_symbol}/{year}/{month_text}/Exness_{archive_symbol}_{year}_{month_text}.zip"


def normalize_exness_tick_csv(csv_bytes: bytes, symbol: str) -> pd.DataFrame:
    """Normalize an Exness archive CSV into SlyTrade's canonical tick schema."""
    raw = pd.read_csv(BytesIO(csv_bytes))
    if raw.empty:
        return pd.DataFrame(columns=TICK_COLUMNS)

    lower_map = {column.lower().strip(): column for column in raw.columns}
    timestamp_column = lower_map.get("timestamp") or lower_map.get("time") or lower_map.get("time_msc")
    bid_column = lower_map.get("bid")
    ask_column = lower_map.get("ask")
    if timestamp_column is None or bid_column is None or ask_column is None:
        raise ValueError(f"Exness CSV missing timestamp/bid/ask columns. Found: {list(raw.columns)}")

    time_msc = pd.to_datetime(raw[timestamp_column], utc=True, errors="coerce")
    frame = pd.DataFrame(
        {
            "time": time_msc.dt.floor("s"),
            "time_msc": time_msc,
            "symbol": normalize_exness_symbol(symbol),
            "bid": pd.to_numeric(raw[bid_column], errors="coerce"),
            "ask": pd.to_numeric(raw[ask_column], errors="coerce"),
            "last": 0.0,
            "volume": 0.0,
            "volume_real": 0.0,
            "flags": 0.0,
        }
    )
    frame = frame.dropna(subset=["time_msc", "bid", "ask"])
    frame["spread"] = frame["ask"] - frame["bid"]
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
    return frame[TICK_COLUMNS].sort_values("time_msc").reset_index(drop=True)


class ExnessArchiveDownloader:
    def __init__(self, output_dir: str | Path = "data/raw", *, base_url: str = EXNESS_TICK_BASE_URL):
        self.output_dir = Path(output_dir)
        self.base_url = base_url

    def month_path(self, symbol: str, month_start: datetime, extension: str = "parquet") -> Path:
        archive_symbol = normalize_exness_symbol(symbol)
        return (
            self.output_dir
            / "exness_ticks"
            / f"symbol={archive_symbol}"
            / f"year={month_start.year:04d}"
            / f"month={month_start.month:02d}"
            / f"period={month_start.year:04d}-{month_start.month:02d}.{extension}"
        )

    def write_month(self, symbol: str, month_start: datetime, frame: pd.DataFrame) -> ExnessArchiveFile:
        path = self.month_path(symbol, month_start)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(path, index=False)
            return ExnessArchiveFile(path=path, rows=len(frame), format="parquet", period=f"{month_start.year:04d}-{month_start.month:02d}")
        except Exception:
            csv_path = path.with_suffix(".csv")
            frame.to_csv(csv_path, index=False)
            return ExnessArchiveFile(path=csv_path, rows=len(frame), format="csv", period=f"{month_start.year:04d}-{month_start.month:02d}")

    def _download_zip_bytes(self, url: str, *, timeout: int = 60) -> bytes:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310 - trusted public data archive URL assembled above
            return response.read()

    def download_month(self, symbol: str, month_start: datetime, *, timeout: int = 60) -> pd.DataFrame:
        archive_symbol = normalize_exness_symbol(symbol)
        url = build_exness_month_url(archive_symbol, month_start.year, month_start.month, base_url=self.base_url)
        zip_bytes = self._download_zip_bytes(url, timeout=timeout)
        with ZipFile(BytesIO(zip_bytes)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV file found inside Exness archive: {url}")
            with archive.open(csv_names[0]) as csv_file:
                return normalize_exness_tick_csv(csv_file.read(), archive_symbol)

    def collect(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        continue_on_error: bool = True,
        timeout: int = 60,
    ) -> ExnessArchiveResult:
        start = ensure_utc(start)
        end = ensure_utc(end)
        archive_symbol = normalize_exness_symbol(symbol)
        rows = 0
        files: list[ExnessArchiveFile] = []
        errors: list[str] = []
        months_attempted = 0
        empty_months = 0
        failed_months = 0

        for month_start in iter_month_starts(start, end):
            months_attempted += 1
            try:
                frame = self.download_month(archive_symbol, month_start, timeout=timeout)
                frame = frame[(frame["time_msc"] >= start) & (frame["time_msc"] < end)].copy()
                if frame.empty:
                    empty_months += 1
                    continue
                written = self.write_month(archive_symbol, month_start, frame)
                rows += written.rows
                files.append(written)
            except Exception as exc:
                failed_months += 1
                message = f"{month_start.year:04d}-{month_start.month:02d}: {exc}"
                errors.append(message)
                if not continue_on_error:
                    raise

        return ExnessArchiveResult(
            symbol=archive_symbol,
            rows=rows,
            files=files,
            months_attempted=months_attempted,
            empty_months=empty_months,
            failed_months=failed_months,
            errors=errors,
        )
