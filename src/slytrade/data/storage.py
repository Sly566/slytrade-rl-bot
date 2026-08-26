from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
    content_hash: str = ""


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
        output = preferred_path
        output_format = "csv"
        if preferred_path.suffix == ".parquet":
            existing_path = preferred_path if preferred_path.exists() else preferred_path.with_suffix(".csv")
            df = self._merge_existing(df, existing_path)
            try:
                payload = self._serialize_parquet(df)
                output_format = "parquet"
            except (ImportError, ModuleNotFoundError):
                output = preferred_path.with_suffix(".csv")
                payload = self._serialize_csv(df)
        else:
            df = self._merge_existing(df, preferred_path)
            payload = self._serialize_csv(df)
        self._atomic_write(output, payload)
        content_hash = hashlib.sha256(payload).hexdigest()
        manifest = {
            "path": str(output),
            "rows": len(df),
            "format": output_format,
            "sha256": content_hash,
            "columns": list(df.columns),
        }
        self._atomic_write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
        return WriteResult(output, len(df), output_format, content_hash)

    @staticmethod
    def _merge_existing(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        if not path.exists():
            return frame
        if path.suffix == ".parquet":
            existing = pd.read_parquet(path)
        elif path.suffix == ".csv":
            existing = pd.read_csv(path)
        else:
            return frame
        combined = pd.concat([existing, frame], ignore_index=True)
        if {"time_msc", "bid", "ask"}.issubset(combined.columns):
            keys = [column for column in ["time_msc", "bid", "ask", "last"] if column in combined.columns]
        elif {"time", "symbol", "timeframe"}.issubset(combined.columns):
            keys = ["time", "symbol", "timeframe"]
        else:
            return frame
        return combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    @staticmethod
    def _serialize_parquet(df: pd.DataFrame) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
            df.to_parquet(handle.name, index=False)
            handle.seek(0)
            return handle.read()

    @staticmethod
    def _serialize_csv(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
        MarketDataStorage._atomic_write(
            path,
            (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
        )

    def write_ticks(self, symbol: str, start: datetime, df: pd.DataFrame) -> WriteResult:
        return self.write_frame(df, self.tick_path(symbol, start))

    def write_bars(self, symbol: str, timeframe: str, start: datetime, df: pd.DataFrame) -> WriteResult:
        return self.write_frame(df, self.bar_path(symbol, timeframe, start))


# ---------------------------------------------------------------------------
# Module-level helpers used by Layer 3/4 pipeline (process/align/scan)
# ---------------------------------------------------------------------------
def discover_partitions(root: Path, pattern: str = "**/*.parquet") -> list[Path]:
    """Return sorted list of parquet files under `root` matching `pattern`."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.glob(pattern))


def _normalize_for_parquet(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Normalise a DataFrame for parquet writing (ensure UTC datetime, reset index)."""
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.reset_index(drop=True)
    return df


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a DataFrame as parquet to `path` (write to tmp, rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    import os
    os.replace(tmp, path)


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def bar_partition(root: Path, kind: str, symbol: str, timeframe: str, year: int, month: int) -> Path:
    """Return partitioned directory for bar data."""
    root = Path(root)
    return root / f"symbol={symbol}" / f"timeframe={timeframe}" / f"year={year}" / f"month={month:02d}"
