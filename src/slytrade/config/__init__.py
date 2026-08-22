"""Configuration models for SlyTrade."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


# Default timeframes for data collection (M1 as execution TF, plus standard HTFs)
DEFAULT_TIMEFRAMES: List[str] = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

# Broker symbol suffixes that MT5 may append (handles XAUUSDm, XAUUSD.a, etc.)
BROKER_SUFFIXES: List[str] = ["", "m", ".m", ".a", ".r", "c", "-micro", "micro", "!"]


class MT5Config(BaseModel):
    """MT5 bridge connection settings."""
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=18812)
    timeout: int = Field(default=60)
    initialize_retry_seconds: float = Field(default=5.0)
    # If None, auto-detects from MT5 terminal info
    account: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None


class DataConfig(BaseModel):
    """Raw data paths."""
    root: Path = Field(default=Path("data/raw"))
    # Subdirectories under root
    mt5_bars_dir: str = "mt5_bars"
    mt5_ticks_dir: str = "mt5_ticks"
    exness_ticks_dir: str = "exness_ticks"
    merged_ticks_dir: str = "merged_ticks"

    @property
    def mt5_bars_path(self) -> Path:
        return self.root / self.mt5_bars_dir

    @property
    def mt5_ticks_path(self) -> Path:
        return self.root / self.mt5_ticks_dir

    @property
    def exness_ticks_path(self) -> Path:
        return self.root / self.exness_ticks_dir

    @property
    def merged_ticks_path(self) -> Path:
        return self.root / self.merged_ticks_dir


class CollectionConfig(BaseModel):
    """Top-level collection run config."""
    symbol: str = "XAUUSD"
    raw_bar_symbol: Optional[str] = None  # e.g. "XAUUSDm" — auto-detected if None
    raw_tick_symbol: Optional[str] = None
    timeframes: List[str] = Field(default_factory=lambda: list(DEFAULT_TIMEFRAMES))
    lookback_years: float = 2.0
    source: str = "hybrid"  # "mt5", "exness", "hybrid"
    clean: bool = False
    # Exness archive settings
    exness_timeout_seconds: int = 120
    exness_retries: int = 1
    exness_retry_backoff: tuple[float, float] = (2.0, 5.0)
    # Tick early-stop: N consecutive empty days = history cliff
    tick_empty_streak_stop: int = 30
    # Progress reporting cadence
    progress_every_files: int = 20
    progress_every_chunks: int = 30


class AppConfig(BaseModel):
    mt5: MT5Config = Field(default_factory=MT5Config)
    data: DataConfig = Field(default_factory=DataConfig)
    collection: CollectionConfig = Field(default_factory=CollectionConfig)
