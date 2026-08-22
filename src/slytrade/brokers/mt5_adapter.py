"""MT5 adapter with safe initialize-per-call and retry logic.

Connects to the Wine-bridged MT5 terminal via RPyC on 127.0.0.1:18812
(launched by start_mt5_bridge.sh). We use mt5linux==1.0.11 which talks to
that RPyC server and exposes the same API surface as the native `MetaTrader5`
Python package.

CRITICAL: mt5.initialize() must be called BEFORE every operation. The
constructor does NOT do this, and the terminal can disconnect at any time.
If initialize() fails, sleep `retry_seconds` and retry exactly once.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import BROKER_SUFFIXES, MT5Config

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Lightweight container for MT5 symbol info (cross-platform)."""
    name: str
    point: float
    digits: int
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_calc_mode: int
    currency_profit: str
    currency_base: str
    currency_margin: str
    spread: int
    trade_contract_size: float = 0.0


def _require_initialized(mt5, timeout: int = 60, retry_seconds: float = 5.0) -> bool:
    """Call mt5.initialize(), retry once after `retry_seconds` on failure.

    Returns True if initialized successfully.
    """
    for attempt in range(2):
        try:
            ok = mt5.initialize(timeout=timeout)
            if ok:
                return True
        except Exception as e:
            logger.warning(f"mt5.initialize() attempt {attempt+1} raised: {e}")
        if attempt == 0:
            logger.info(f"mt5.initialize() failed, retrying in {retry_seconds}s...")
            time.sleep(retry_seconds)
    return False


class MT5Client:
    """Wrapper around mt5linux that handles initialize/retry/shutdown."""

    def __init__(self, config: Optional[MT5Config] = None):
        try:
            from mt5linux import MetaTrader5  # type: ignore
        except ImportError as e:
            raise ImportError(
                "mt5linux is required. Install with: pip install mt5linux==1.0.11"
            ) from e

        self.config = config or MT5Config()
        self.mt5 = MetaTrader5(
            host=self.config.host,
            port=self.config.port,
        )
        self._symbol_cache: Dict[str, SymbolInfo] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """Explicit initialize (user can call this to warm up; all other
        methods call it automatically before every operation)."""
        return _require_initialized(
            self.mt5,
            timeout=self.config.timeout,
            retry_seconds=self.config.initialize_retry_seconds,
        )

    def shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    def terminal_info(self) -> Any:
        if not self.initialize():
            raise ConnectionError("MT5 initialize failed — is start_mt5_bridge.sh running?")
        info = self.mt5.terminal_info()
        if info is None:
            raise ConnectionError("MT5 terminal_info() returned None")
        return info

    def account_info(self) -> Any:
        if not self.initialize():
            raise ConnectionError("MT5 initialize failed")
        info = self.mt5.account_info()
        if info is None:
            raise ConnectionError("MT5 account_info() returned None")
        return info

    # ------------------------------------------------------------------
    # Symbol discovery
    # ------------------------------------------------------------------
    def _to_symbol_info(self, raw: Any) -> SymbolInfo:
        return SymbolInfo(
            name=raw.name,
            point=raw.point,
            digits=raw.digits,
            contract_size=getattr(raw, "trade_contract_size", 100000.0),
            volume_min=getattr(raw, "volume_min", 0.01),
            volume_max=getattr(raw, "volume_max", 100.0),
            volume_step=getattr(raw, "volume_step", 0.01),
            trade_calc_mode=getattr(raw, "trade_calc_mode", 0),
            currency_profit=getattr(raw, "currency_profit", "USD"),
            currency_base=getattr(raw, "currency_base", ""),
            currency_margin=getattr(raw, "currency_margin", ""),
            spread=getattr(raw, "spread", 0),
        )

    def symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        if not self.initialize():
            return None
        raw = self.mt5.symbol_info(symbol)
        if raw is None:
            return None
        if not getattr(raw, "visible", True):
            try:
                self.mt5.symbol_select(symbol, True)
            except Exception:
                pass
        info = self._to_symbol_info(raw)
        self._symbol_cache[symbol] = info
        return info

    def detect_raw_symbol(self, base_symbol: str, prefer: Optional[str] = None) -> Optional[str]:
        """Find the broker-suffixed symbol name on the terminal (e.g. XAUUSDm)."""
        from .symbols import symbol_variants
        candidates: List[str] = []
        if prefer:
            candidates.append(prefer)
        candidates.extend(symbol_variants(base_symbol))
        for sym in candidates:
            info = self.symbol_info(sym)
            if info is not None:
                return sym
        return None

    def all_symbols(self) -> List[SymbolInfo]:
        if not self.initialize():
            return []
        raw = self.mt5.symbols_get()
        if raw is None:
            return []
        out = []
        for r in raw:
            try:
                out.append(self._to_symbol_info(r))
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for [start, end] inclusive. Returns DataFrame
        with columns: time, open, high, low, close, tick_volume, spread, real_volume
        """
        if not self.initialize():
            raise ConnectionError(f"MT5 initialize failed — cannot fetch bars for {symbol} {timeframe}")

        # mt5linux exposes the native MetaTrader5 constants on its own object
        tf_const = getattr(self.mt5, f"TIMEFRAME_{timeframe}", None)
        if tf_const is None:
            raise ValueError(f"Unknown MT5 timeframe: {timeframe}")

        start_dt = pd.Timestamp(start).to_pydatetime()
        end_dt = pd.Timestamp(end).to_pydatetime()
        # copy_rates_from counts BACKWARD from a date; we use copy_rates_range
        rates = self.mt5.copy_rates_range(symbol, tf_const, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        return df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]

    def get_ticks(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        flags: int = 0,  # COPY_TICKS_ALL = 0
    ) -> pd.DataFrame:
        """Fetch ticks for [start, end] inclusive. Returns DataFrame with
        columns: time_msc, bid, ask, last, volume, flags.
        """
        if not self.initialize():
            raise ConnectionError(f"MT5 initialize failed — cannot fetch ticks for {symbol}")

        # mt5linux exposes COPY_TICKS_ALL etc. directly on its client object.
        # COPY_TICKS_ALL = 0 per MT5 API spec.
        tick_flag = flags if flags else getattr(self.mt5, "COPY_TICKS_ALL", 0)
        start_dt = pd.Timestamp(start).to_pydatetime()
        end_dt = pd.Timestamp(end).to_pydatetime()

        ticks = self.mt5.copy_ticks_range(symbol, start_dt, end_dt, tick_flag)
        if ticks is None or len(ticks) == 0:
            cols = ["time_msc", "bid", "ask", "last", "volume", "flags"]
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(ticks)
        # time_msc is milliseconds since epoch
        df["time_msc"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        df = df.sort_values("time_msc").drop_duplicates(subset=["time_msc"]).reset_index(drop=True)
        keep = ["time_msc", "bid", "ask", "last", "volume", "flags"]
        for c in keep:
            if c not in df.columns:
                df[c] = 0.0
        return df[keep]
