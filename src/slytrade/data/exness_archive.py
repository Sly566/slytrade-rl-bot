"""Exness public tick-archive downloader.

URL format:
    https://ticks.ex2archive.com/ticks/{SYMBOL}/{YYYY}/{MM}/Exness_{SYMBOL}_{YYYY}_{MM}.zip

Each zip contains one CSV with columns (timestamp, bid, ask, last, volume,
flags, spread, mid). Timestamps are ISO-8601 with a trailing ``Z``
(``2026-08-02 22:01:30.645Z``); we strip the Z and parse explicitly so pandas
uses the fast vectorised path instead of per-element dateutil.

Downloads are streaming-chunked into snappy parquet files under
``data/raw/exness_ticks/symbol=SYM/year=YYYY/month=MM/part-0.parquet`` to
keep peak memory at O(chunk) instead of O(month) (one month of XAUUSD ticks
is ~400 MB CSV / 90M rows). Idempotent: already-downloaded months are skipped
unless ``skip_existing=False``.
"""
from __future__ import annotations

import io
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from ..config import DataConfig
from .storage import tick_month_partition, write_partition
from .time import ensure_utc, iter_months


ProgressFn = Callable[[str], None]

EXNESS_BASE = "https://ticks.ex2archive.com/ticks"
USER_AGENT = "Mozilla/5.0 (compatible; SlyTrade/0.3; +https://github.com/Sly566/slytrade-rl-bot)"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_symbol(symbol: str) -> str:
    """Strip broker suffix for archive URL (e.g. XAUUSDm -> XAUUSD)."""
    s = symbol.strip().upper()
    # Conservative: only drop a single trailing lowercase m (Exness CFD suffix).
    if s.endswith("M") and len(s) > 6 and s[-1] == "M":
        return s[:-1]
    return s


def build_url(symbol: str, year: int, month: int, *, base: str = EXNESS_BASE) -> str:
    sym = normalize_symbol(symbol)
    return f"{base.rstrip('/')}/{sym}/{year:04d}/{month:02d}/Exness_{sym}_{year:04d}_{month:02d}.zip"


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse Exness timestamps fast (Z-strip + explicit format)."""
    if values.empty:
        return pd.to_datetime(values, utc=True, errors="coerce")
    # If already datetime, just coerce to UTC.
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return pd.to_datetime(values, utc=True, errors="coerce")
    head = values.iloc[0]
    total = len(values)
    if isinstance(head, str) and head.rstrip().endswith(("Z", "z")):
        stripped = values.str.rstrip("Zz")
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            parsed = pd.to_datetime(stripped, utc=True, errors="coerce", format=fmt)
            if float(parsed.notna().sum()) >= total * 0.99:
                return parsed
    # Fallback: explicit formats then slow inference.
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y.%m.%d %H:%M:%S.%f",
        "%Y.%m.%d %H:%M:%S",
    ):
        parsed = pd.to_datetime(values, utc=True, errors="coerce", format=fmt)
        if float(parsed.notna().sum()) >= total * 0.99:
            return parsed
    return pd.to_datetime(values, utc=True, errors="coerce")


def _chunk_to_canonical(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in raw.columns}
    t_col = lower.get("timestamp") or lower.get("time") or lower.get("time_msc")
    b_col = lower.get("bid")
    a_col = lower.get("ask")
    if t_col is None or b_col is None or a_col is None:
        raise ValueError(f"Exness CSV missing timestamp/bid/ask. Found {list(raw.columns)}")

    time_msc = parse_timestamps(raw[t_col])
    out = pd.DataFrame(
        {
            "time_msc": time_msc,
            "bid": pd.to_numeric(raw[b_col], errors="coerce"),
            "ask": pd.to_numeric(raw[a_col], errors="coerce"),
            "last": pd.to_numeric(raw.get(lower.get("last", ""), 0.0), errors="coerce")
            if "last" in lower
            else 0.0,
            "volume": pd.to_numeric(raw.get(lower.get("volume", ""), 0.0), errors="coerce")
            if "volume" in lower
            else 0.0,
            "flags": 0,
        }
    )
    out = out.dropna(subset=["time_msc", "bid", "ask"])
    out["spread"] = out["ask"] - out["bid"]
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["symbol"] = normalize_symbol(symbol)
    keep = ["time_msc", "bid", "ask", "last", "volume", "flags", "spread", "mid", "symbol"]
    return out[keep]


# --------------------------------------------------------------------------- #
# Downloader
# --------------------------------------------------------------------------- #
@dataclass
class ExnessResult:
    total_rows: int = 0
    files: int = 0
    empty_months: int = 0
    failed_months: int = 0
    errors: list[str] = field(default_factory=list)


class ExnessArchiveDownloader:
    """Download and normalise monthly tick archives from Exness."""

    def __init__(
        self,
        data_cfg: DataConfig,
        *,
        timeout: int = 120,
        retries: int = 1,
        retry_backoff: tuple[float, float] = (2.0, 5.0),
        skip_existing: bool = True,
        progress: ProgressFn | None = None,
        chunksize: int = 500_000,
    ):
        self.data = data_cfg
        self.timeout = timeout
        self.retries = retries
        self.backoff = retry_backoff
        self.skip_existing = skip_existing
        self.progress = progress or (lambda _m: None)
        self.chunksize = chunksize

    # ------------------------------------------------------------------ #
    def _http_get(self, url: str, label: str) -> bytes:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = _time.monotonic()
            try:
                req = Request(url, headers={"User-Agent": USER_AGENT})
                with urlopen(req, timeout=self.timeout) as r:  # noqa: S310
                    data = r.read()
                dt = _time.monotonic() - t0
                mb = len(data) / (1024 * 1024)
                self.progress(
                    f"    {label} downloaded {mb:.1f} MB in {dt:.1f}s "
                    f"({mb / max(dt, 0.1):.1f} MB/s)"
                )
                return data
            except Exception as e:
                last_err = e
                self.progress(f"    {label} attempt {attempt + 1} failed: {e}")
                if attempt < self.retries:
                    _time.sleep(self.backoff[0] + attempt * (self.backoff[1] - self.backoff[0]))
        assert last_err is not None
        raise last_err

    # ------------------------------------------------------------------ #
    def _download_month_stream(self, symbol: str, year: int, month: int) -> pd.DataFrame | None:
        label = f"{year:04d}-{month:02d}"
        sym = normalize_symbol(symbol)
        url = build_url(sym, year, month)
        part_dir = tick_month_partition(self.data.exness_ticks_path, "", sym, year, month)
        target = part_dir / "part-0.parquet"

        if self.skip_existing and target.exists() and target.stat().st_size > 4096:
            try:
                existing = pd.read_parquet(target)
                if len(existing) > 0:
                    self.progress(
                        f"    {label} already exists ({len(existing):,} rows) — skipping."
                    )
                    return existing
            except Exception:
                pass  # corrupt file — redownload

        self.progress(f"    downloading {label} ...")
        try:
            zbytes = self._http_get(url, label)
        except Exception as e:
            raise RuntimeError(f"{label}: {e}") from e

        parse_t0 = _time.monotonic()
        total = 0
        frames: list[pd.DataFrame] = []
        try:
            with ZipFile(BytesIO(zbytes)) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    self.progress(f"    {label} no CSV in zip")
                    return None
                with zf.open(csv_names[0]) as fh:
                    wrapper = io.TextIOWrapper(fh, encoding="utf-8")
                    for chunk in pd.read_csv(wrapper, dtype=str, chunksize=self.chunksize):
                        c = _chunk_to_canonical(chunk, sym)
                        if not c.empty:
                            frames.append(c)
                            total += len(c)
                            if total % (self.chunksize * 4) < self.chunksize:
                                elapsed = _time.monotonic() - parse_t0
                                self.progress(
                                    f"    {label} parsed {total:,} rows "
                                    f"({total / max(elapsed, 0.1):,.0f} rows/s)"
                                )
        except Exception as e:
            raise RuntimeError(f"{label} parse error: {e}") from e

        if not frames:
            self.progress(f"    {label} empty")
            return None

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values("time_msc", kind="mergesort").drop_duplicates(
            subset=["time_msc"], keep="last"
        ).reset_index(drop=True)

        dt = _time.monotonic() - parse_t0
        self.progress(
            f"    {label} ... ✓ {len(out):,} rows in {dt:.1f}s "
            f"({len(out) / max(dt, 0.1):,.0f} rows/s)"
        )
        return out

    # ------------------------------------------------------------------ #
    def collect_range(
        self,
        *,
        symbol: str,
        start: date | datetime,
        end: date | datetime,
    ) -> ExnessResult:
        if isinstance(start, datetime):
            start = ensure_utc(start).date()
        if isinstance(end, datetime):
            end = ensure_utc(end).date()

        result = ExnessResult()
        months = list(iter_months(start, end))
        if not months:
            return result

        sym = normalize_symbol(symbol)
        self.progress(f"  Exness archive: {len(months)} month(s) to fetch "
                      f"({months[0][0].isoformat()[:7]} → {months[-1][0].isoformat()[:7]})")

        for m_start, _m_end in months:
            y, mo = m_start.year, m_start.month
            label = f"{y:04d}-{mo:02d}"
            try:
                df = self._download_month_stream(symbol, y, mo)
                if df is None or df.empty:
                    result.empty_months += 1
                    continue
                part_dir = tick_month_partition(self.data.exness_ticks_path, "", sym, y, mo)
                write_partition(df, part_dir, time_col="time_msc")
                result.total_rows += len(df)
                result.files += 1
            except Exception as e:
                result.failed_months += 1
                result.errors.append(f"{label}: {e}")
                self.progress(f"    ! {label} FAILED: {e}")
        return result
