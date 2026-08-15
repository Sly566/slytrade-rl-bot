"""Memory-bounded tick processing for arbitrarily large lookbacks.

The in-memory ``align_market_data`` materialises the full tick set several times
(sort, per-bar features, decision quotes, coverage), which OOMs on long
lookbacks (1y of XAUUSD ≈ 70M ticks ≈ multiple GB). This module streams
month-partitioned tick files instead: it holds at most one file chunk plus one
bar's ticks in memory, and computes everything in a single ascending pass.

The outputs are numerically identical to the in-memory path (verified by the
parity tests).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from slytrade.data.alignment import (
    TICK_BAR_FEATURE_COLUMNS,
    AlignedDataset,
    DatasetManifest,
    attach_ict_features,
    compute_fresh_coverage_ratio,
    dataset_quality_status,
    infer_canonical_symbol,
    infer_single_symbol,
    infer_single_timeframe,
)
from slytrade.data.diagnostics import TickBarCoverageDiagnostics
from slytrade.data.exness_archive import normalize_exness_symbol
from slytrade.data.schemas import BAR_COLUMNS, TICK_COLUMNS
from slytrade.data.timeframes import add_decision_time

_TICK_READ_COLUMNS = ("time_msc", "bid", "ask")


def _file_sort_key(path: Path) -> tuple[int, int, int, str]:
    """Order tick files chronologically by their encoded year/month/day."""
    year = month = day = 0
    for part in path.parts:
        if part.startswith("year="):
            year = int(part.split("=", 1)[1])
        elif part.startswith("month="):
            month = int(part.split("=", 1)[1])
        elif part.startswith("day="):
            day = int(part.split("=", 1)[1])
        elif part.startswith("period="):  # exness period=YYYY-MM
            value = part.split("=", 1)[1].split(".")[0]
            year, month = (int(bit) for bit in value.split("-")[:2])
    return (year, month, day, str(path))


def sort_tick_files(files: list[Path]) -> list[Path]:
    return sorted(files, key=_file_sort_key)


def read_tick_columns(path: Path) -> pd.DataFrame:
    """Read only the columns the streaming pass needs (time_msc, bid, ask)."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=list(_TICK_READ_COLUMNS))
    return pd.read_csv(path, usecols=list(_TICK_READ_COLUMNS))


def _to_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, utc=True).astype("int64").to_numpy()


class TickCursor:
    """Ascending tick iterator across many files, holding one file at a time."""

    def __init__(self, files: list[Path]):
        self._files = files
        self._index = 0
        self._times: np.ndarray | None = None
        self._bids: np.ndarray | None = None
        self._asks: np.ndarray | None = None
        self._pos = 0
        self._load_next()

    def _load_next(self) -> None:
        while self._index < len(self._files):
            frame = read_tick_columns(self._files[self._index])
            self._index += 1
            if frame.empty:
                continue
            times = _to_ns(frame["time_msc"])
            bids = pd.to_numeric(frame["bid"], errors="coerce").to_numpy(dtype=float)
            asks = pd.to_numeric(frame["ask"], errors="coerce").to_numpy(dtype=float)
            # Files are normally pre-sorted; sort only when needed.
            if len(times) > 1 and not bool(np.all(np.diff(times) >= 0)):
                order = np.argsort(times, kind="stable")
                times, bids, asks = times[order], bids[order], asks[order]
            self._times, self._bids, self._asks, self._pos = times, bids, asks, 0
            return
        self._times = self._bids = self._asks = None
        self._pos = 0

    def peek(self) -> tuple[int, float, float] | None:
        while self._times is not None and self._pos >= len(self._times):
            self._load_next()
        times = self._times
        bids = self._bids
        asks = self._asks
        if times is None or bids is None or asks is None:
            return None
        return int(times[self._pos]), float(bids[self._pos]), float(asks[self._pos])

    def advance(self) -> None:
        self._pos += 1


class TickFileMeta(TypedDict):
    rows: int
    start: str
    end: str
    symbol: str


def tick_file_metadata(files: list[Path]) -> TickFileMeta:
    """Cheap metadata over a set of tick files: rows, first/last tick, symbol."""
    ordered = sort_tick_files(files)
    if not ordered:
        raise ValueError("no tick files provided")

    total_rows = 0
    for path in ordered:
        if path.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq

                total_rows += int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:  # pragma: no cover - fall back to reading
                total_rows += len(read_tick_columns(path))
        else:
            total_rows += len(read_tick_columns(path))

    def _minmax_time(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
        # Read only the timestamp column (cheap) instead of the full frame.
        if path.suffix.lower() == ".parquet":
            column = pd.read_parquet(path, columns=["time_msc"])["time_msc"]
        else:
            column = pd.read_csv(path, usecols=["time_msc"])["time_msc"]
        times = pd.to_datetime(column, utc=True)
        return times.min(), times.max()

    start_ts, _ = _minmax_time(ordered[0])
    _, end_ts = _minmax_time(ordered[-1])

    symbol = "UNKNOWN"
    if ordered[0].suffix.lower() == ".parquet":
        extra = pd.read_parquet(ordered[0], columns=["symbol"])
    else:
        extra = pd.read_csv(ordered[0], usecols=["symbol"])
    if extra["symbol"].notna().any():
        symbol = str(extra["symbol"].dropna().iloc[0])

    return {"rows": total_rows, "start": str(start_ts), "end": str(end_ts), "symbol": normalize_exness_symbol(symbol)}


def align_market_data_streaming(
    bars: pd.DataFrame,
    tick_files: list[Path],
    *,
    timeframe: str | None = None,
    canonical_symbol: str | None = None,
    bar_source: str = "mt5_bars",
    tick_source: str = "exness_ticks",
    max_quote_age_seconds: float = 5.0,
    min_fresh_coverage: float = 0.95,
    include_ict_features: bool = True,
    require_fresh_quotes: bool = False,
) -> AlignedDataset:
    """Align bars against partitioned tick files without loading all ticks.

    One ascending pass computes, per bar: tick microstructure features, the
    decision quote (last tick at/before decision time) and the freshness
    coverage — holding at most one file + one bar's ticks in memory.
    """
    if bars.empty:
        raise ValueError("bars cannot be empty")
    ordered_files = sort_tick_files(tick_files)
    if not ordered_files:
        raise ValueError("no tick files provided")
    meta = tick_file_metadata(ordered_files)

    bar_symbol = infer_single_symbol(bars)
    resolved_timeframe = (timeframe or infer_single_timeframe(bars)).upper()
    resolved_canonical = infer_canonical_symbol(bar_symbol, str(meta["symbol"]), canonical_symbol)

    aligned_bars = add_decision_time(bars, timeframe=resolved_timeframe).sort_values("time").reset_index(drop=True)
    aligned_bars["time"] = pd.to_datetime(aligned_bars["time"], utc=True)
    aligned_bars["decision_time"] = pd.to_datetime(aligned_bars["decision_time"], utc=True)
    aligned_bars["symbol"] = resolved_canonical
    if include_ict_features:
        aligned_bars = attach_ict_features(aligned_bars)

    bar_times = _to_ns(aligned_bars["time"])
    decision_times = _to_ns(aligned_bars["decision_time"])

    features, quotes, coverage = _stream_features_quotes_coverage(
        bar_times, decision_times, ordered_files, max_quote_age_seconds
    )

    for column in TICK_BAR_FEATURE_COLUMNS:
        aligned_bars[column] = features[column]
    aligned_bars["quote_time"] = quotes["quote_time"]
    aligned_bars["quote_bid"] = quotes["quote_bid"]
    aligned_bars["quote_ask"] = quotes["quote_ask"]
    aligned_bars["quote_mid"] = quotes["quote_mid"]
    aligned_bars["quote_spread"] = quotes["quote_spread"]
    aligned_bars["quote_age_seconds"] = quotes["quote_age_seconds"]
    aligned_bars["quote_is_fresh"] = quotes["quote_is_fresh"]

    fresh_ratio = compute_fresh_coverage_ratio(coverage)
    quality_status, quality_issues = dataset_quality_status(coverage, min_fresh_coverage=min_fresh_coverage)
    source_bars_rows = len(aligned_bars)
    if require_fresh_quotes:
        aligned_bars = aligned_bars[aligned_bars["quote_is_fresh"]].reset_index(drop=True)
    aligned_bars_rows = len(aligned_bars)
    dropped_stale_bars = source_bars_rows - aligned_bars_rows
    if aligned_bars.empty:
        raise ValueError("aligned dataset has no bars after fresh-quote filtering")

    tick_dir = ordered_files[0].parent
    while tick_dir.name.startswith("year=") or tick_dir.name.startswith("month=") or tick_dir.name.startswith("day=") or tick_dir.name.startswith("period="):
        tick_dir = tick_dir.parent

    manifest = DatasetManifest(
        canonical_symbol=resolved_canonical,
        bar_symbol=bar_symbol,
        tick_symbol=str(meta["symbol"]),
        bar_source=bar_source,
        tick_source=tick_source,
        timeframe=resolved_timeframe,
        bars_rows=len(aligned_bars),
        ticks_rows=meta["rows"],
        bars_start=str(aligned_bars["time"].min()),
        bars_end=str(aligned_bars["time"].max()),
        decision_start=str(aligned_bars["decision_time"].min()),
        decision_end=str(aligned_bars["decision_time"].max()),
        ticks_start=str(meta["start"]),
        ticks_end=str(meta["end"]),
        coverage=coverage.__dict__,
        fresh_coverage_ratio=fresh_ratio,
        quality_status=quality_status,
        quality_issues=quality_issues,
        source_bars_rows=source_bars_rows,
        aligned_bars_rows=aligned_bars_rows,
        dropped_stale_bars=dropped_stale_bars,
        require_fresh_quotes=require_fresh_quotes,
        source_files={"bars": "", "ticks": str(tick_dir)},
    )
    return AlignedDataset(bars=aligned_bars, ticks=pd.DataFrame(columns=TICK_COLUMNS), manifest=manifest)


def _stream_features_quotes_coverage(
    bar_times: np.ndarray,
    decision_times: np.ndarray,
    files: list[Path],
    max_age_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, TickBarCoverageDiagnostics]:
    n = len(bar_times)

    tick_count = np.zeros(n, dtype=float)
    tick_rate = np.zeros(n, dtype=float)
    spread_mean = np.zeros(n, dtype=float)
    spread_max = np.zeros(n, dtype=float)
    mid_open = np.zeros(n, dtype=float)
    mid_high = np.zeros(n, dtype=float)
    mid_low = np.zeros(n, dtype=float)
    mid_close = np.zeros(n, dtype=float)
    mid_range = np.zeros(n, dtype=float)
    mid_return = np.zeros(n, dtype=float)

    quote_time_ns = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    quote_bid = np.full(n, np.nan, dtype=float)
    quote_ask = np.full(n, np.nan, dtype=float)

    cursor = TickCursor(files)
    current = cursor.peek()
    seen = 0
    fresh = 0
    stale = 0
    max_observed_age = 0.0
    first_missing: str | None = None
    first_stale: str | None = None
    last_tick: tuple[int, float, float] | None = None
    carry: deque[tuple[int, float, float]] = deque()

    for i in range(n):
        bar_t = int(bar_times[i])
        dec_t = int(decision_times[i])
        # A tick exactly at the bar boundary is counted in BOTH adjacent bars
        # by the in-memory path (<= decision and >= next bar open); carry those.
        window: deque[tuple[int, float, float]] = deque(carry)
        carry = deque()

        while current is not None and current[0] <= dec_t:
            if current[0] >= bar_t:
                window.append(current)
            if current[0] == dec_t:
                carry.append(current)
            last_tick = current
            cursor.advance()
            current = cursor.peek()

        count = len(window)
        tick_count[i] = float(count)
        duration = max((dec_t - bar_t) / 1e9, 1e-9)
        tick_rate[i] = count / duration
        if count:
            mids = [(b + a) / 2.0 for (_, b, a) in window]
            spreads = [a - b for (_, b, a) in window]
            mid_open[i] = mids[0]
            mid_close[i] = mids[-1]
            mid_high[i] = max(mids)
            mid_low[i] = min(mids)
            spread_mean[i] = sum(spreads) / count
            spread_max[i] = max(spreads)
            mid_range[i] = mid_high[i] - mid_low[i]
            mid_return[i] = (mid_close[i] - mid_open[i]) / abs(mid_open[i]) if mid_open[i] else 0.0

        if last_tick is not None:
            seen += 1
            age = (dec_t - last_tick[0]) / 1e9
            max_observed_age = max(max_observed_age, age)
            quote_time_ns[i] = last_tick[0]
            quote_bid[i] = last_tick[1]
            quote_ask[i] = last_tick[2]
            if 0.0 <= age <= max_age_seconds:
                fresh += 1
            else:
                stale += 1
                if first_stale is None:
                    first_stale = str(pd.Timestamp(dec_t, tz="UTC"))
        else:
            if first_missing is None:
                first_missing = str(pd.Timestamp(dec_t, tz="UTC"))

    features = pd.DataFrame(
        {
            "tick_count": tick_count,
            "tick_rate_per_second": tick_rate,
            "tick_spread_mean": spread_mean,
            "tick_spread_max": spread_max,
            "tick_mid_open": mid_open,
            "tick_mid_high": mid_high,
            "tick_mid_low": mid_low,
            "tick_mid_close": mid_close,
            "tick_mid_range": mid_range,
            "tick_mid_return": mid_return,
        }
    )

    quotes = pd.DataFrame(
        {
            "quote_time": pd.to_datetime(quote_time_ns, unit="ns", utc=True).where(quote_time_ns != np.iinfo(np.int64).min),
            "quote_bid": quote_bid,
            "quote_ask": quote_ask,
        }
    )
    quotes["quote_mid"] = (quotes["quote_bid"] + quotes["quote_ask"]) / 2.0
    quotes["quote_spread"] = quotes["quote_ask"] - quotes["quote_bid"]
    quotes["quote_age_seconds"] = (pd.to_datetime(decision_times, unit="ns", utc=True) - quotes["quote_time"]).dt.total_seconds()
    quotes["quote_is_fresh"] = (quotes["quote_age_seconds"] >= 0.0) & (quotes["quote_age_seconds"] <= max_age_seconds)
    quotes["quote_is_fresh"] = quotes["quote_is_fresh"].fillna(False)

    coverage = TickBarCoverageDiagnostics(
        bars=n,
        bars_with_tick_before_decision=seen,
        bars_missing_tick_before_decision=n - seen,
        bars_with_fresh_tick_before_decision=fresh,
        bars_with_stale_tick_before_decision=stale,
        max_quote_age_seconds=max_age_seconds,
        max_observed_quote_age_seconds=max_observed_age,
        first_missing_decision_time=first_missing,
        first_stale_decision_time=first_stale,
    )
    return features, quotes, coverage


def resample_ticks_to_bars_streaming(
    tick_files: list[Path],
    timeframe: str,
    *,
    symbol: str,
) -> pd.DataFrame:
    """Build OHLC bars from partitioned tick files in a single streaming pass.

    Buckets ticks by the timeframe calendar grid using bar-open timestamps
    (MT5 convention), keeping only one bucket in memory at a time.
    """
    from slytrade.data.timeframes import timeframe_duration

    normalized_tf = timeframe.upper()
    duration_ns = int(timeframe_duration(normalized_tf).total_seconds() * 1e9)

    rows: list[dict] = []
    cursor = TickCursor(sort_tick_files(tick_files))
    current = cursor.peek()
    bucket_ns: int | None = None
    open_p = high_p = low_p = close_p = 0.0
    tick_vol = 0
    spread_acc = 0.0

    def flush() -> None:
        nonlocal bucket_ns, tick_vol, spread_acc
        if bucket_ns is not None and tick_vol:
            rows.append(
                {
                    "time": pd.Timestamp(bucket_ns, tz="UTC"),
                    "symbol": symbol,
                    "timeframe": normalized_tf,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "tick_volume": float(tick_vol),
                    "spread": spread_acc / tick_vol,
                    "real_volume": 0.0,
                }
            )
        bucket_ns = None
        tick_vol = 0
        spread_acc = 0.0

    while current is not None:
        t_ns, bid, ask = current
        mid = (bid + ask) / 2.0
        this_bucket = (t_ns // duration_ns) * duration_ns
        if bucket_ns is None:
            bucket_ns, open_p, high_p, low_p, close_p = this_bucket, mid, mid, mid, mid
        elif this_bucket != bucket_ns:
            flush()
            bucket_ns, open_p, high_p, low_p, close_p = this_bucket, mid, mid, mid, mid
        else:
            high_p = max(high_p, mid)
            low_p = min(low_p, mid)
            close_p = mid
        tick_vol += 1
        spread_acc += ask - bid
        cursor.advance()
        current = cursor.peek()
    flush()

    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return pd.DataFrame(rows)[BAR_COLUMNS].sort_values("time").reset_index(drop=True)


def merge_tick_sources_streaming(
    exness_files: list[Path],
    mt5_recent: pd.DataFrame,
    *,
    out_root: Path,
    symbol: str,
) -> int:
    """Stream Exness month files into merged storage, folding in recent MT5 ticks.

    Writes one merged month file at a time (memory O(one month + recent)), and
    de-duplicates the overlap between the last Exness month and the recent MT5
    ticks. Returns the total number of merged tick rows.
    """
    ordered = sort_tick_files(exness_files)
    recent = mt5_recent.copy()
    recent["time_msc"] = pd.to_datetime(recent["time_msc"], utc=True)

    total_rows = 0
    last_index = len(ordered) - 1
    for idx, path in enumerate(ordered):
        frame = (
            pd.read_parquet(path)
            if path.suffix.lower() == ".parquet"
            else pd.read_csv(path)
        )
        frame["time_msc"] = pd.to_datetime(frame["time_msc"], utc=True)
        if idx == last_index and not recent.empty:
            frame = pd.concat([frame, recent], ignore_index=True)
            frame = frame.drop_duplicates(subset=["time_msc"], keep="first").sort_values("time_msc").reset_index(drop=True)
        else:
            frame = frame.sort_values("time_msc").reset_index(drop=True)
        frame["symbol"] = symbol
        _write_month_parquet(out_root, symbol, frame)
        total_rows += len(frame)
    return total_rows


def _write_month_parquet(out_root: Path, symbol: str, frame: pd.DataFrame) -> None:
    times = pd.to_datetime(frame["time_msc"], utc=True)
    for (year, month), group in frame.groupby([times.dt.year, times.dt.month]):
        directory = out_root / f"symbol={symbol}" / f"year={year}" / f"month={int(month):02d}"
        directory.mkdir(parents=True, exist_ok=True)
        group.to_parquet(directory / "ticks.parquet", index=False)
