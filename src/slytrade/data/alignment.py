from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from slytrade.data.diagnostics import TickBarCoverageDiagnostics, inspect_tick_bar_coverage
from slytrade.data.exness_archive import normalize_exness_symbol
from slytrade.data.timeframes import add_decision_time
from slytrade.features.ict import FEATURE_COLUMNS, compute_ict_features

QUOTE_COLUMNS = [
    "quote_time",
    "quote_bid",
    "quote_ask",
    "quote_mid",
    "quote_spread",
    "quote_age_seconds",
    "quote_is_fresh",
]

TICK_BAR_FEATURE_COLUMNS = [
    "tick_count",
    "tick_rate_per_second",
    "tick_spread_mean",
    "tick_spread_max",
    "tick_mid_open",
    "tick_mid_high",
    "tick_mid_low",
    "tick_mid_close",
    "tick_mid_range",
    "tick_mid_return",
]


@dataclass(frozen=True)
class DatasetManifest:
    canonical_symbol: str
    bar_symbol: str
    tick_symbol: str
    bar_source: str
    tick_source: str
    timeframe: str
    bars_rows: int
    ticks_rows: int
    bars_start: str
    bars_end: str
    decision_start: str
    decision_end: str
    ticks_start: str
    ticks_end: str
    coverage: dict[str, object]
    fresh_coverage_ratio: float = 0.0
    quality_status: str = "UNKNOWN"
    quality_issues: list[str] = field(default_factory=list)
    quote_columns: list[str] = field(default_factory=lambda: QUOTE_COLUMNS.copy())
    tick_feature_columns: list[str] = field(default_factory=lambda: TICK_BAR_FEATURE_COLUMNS.copy())
    ict_feature_columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS.copy())
    source_files: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class AlignedDataset:
    bars: pd.DataFrame
    ticks: pd.DataFrame
    manifest: DatasetManifest


def infer_single_symbol(frame: pd.DataFrame, *, column: str = "symbol") -> str:
    if column not in frame.columns or frame.empty:
        raise ValueError(f"Cannot infer symbol: missing/empty column {column}")
    symbols = sorted(str(value) for value in frame[column].dropna().unique())
    if len(symbols) != 1:
        raise ValueError(f"Expected exactly one symbol in {column}, found {symbols}")
    return symbols[0]


def infer_single_timeframe(frame: pd.DataFrame) -> str:
    if "timeframe" not in frame.columns or frame.empty:
        raise ValueError("Cannot infer timeframe: missing/empty timeframe column")
    timeframes = sorted(str(value).upper() for value in frame["timeframe"].dropna().unique())
    if len(timeframes) != 1:
        raise ValueError(f"Expected exactly one timeframe, found {timeframes}")
    return timeframes[0]


def infer_canonical_symbol(bar_symbol: str, tick_symbol: str, canonical_symbol: str | None = None) -> str:
    if canonical_symbol:
        return canonical_symbol.strip().upper()
    normalized_bar = normalize_exness_symbol(bar_symbol)
    normalized_tick = normalize_exness_symbol(tick_symbol)
    if normalized_bar == normalized_tick:
        return normalized_bar
    raise ValueError(
        "Could not infer canonical symbol from bar/tick symbols. "
        f"bar_symbol={bar_symbol}, tick_symbol={tick_symbol}. Pass canonical_symbol explicitly."
    )


def compute_fresh_coverage_ratio(coverage: TickBarCoverageDiagnostics) -> float:
    if coverage.bars <= 0:
        return 0.0
    return float(coverage.bars_with_fresh_tick_before_decision / coverage.bars)


def dataset_quality_status(
    coverage: TickBarCoverageDiagnostics,
    *,
    min_fresh_coverage: float = 0.95,
) -> tuple[str, list[str]]:
    issues: list[str] = []
    ratio = compute_fresh_coverage_ratio(coverage)
    if ratio < min_fresh_coverage:
        issues.append(f"fresh quote coverage {ratio:.2%} below required {min_fresh_coverage:.2%}")
    if coverage.bars_missing_tick_before_decision:
        issues.append(f"{coverage.bars_missing_tick_before_decision} bars have no earlier tick")
    if coverage.bars_with_stale_tick_before_decision:
        issues.append(f"{coverage.bars_with_stale_tick_before_decision} bars have stale latest ticks")
    return ("PASS" if not issues else "WARN", issues)


def attach_ict_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Attach causal ICT/SMC features if they are missing."""
    missing = [column for column in FEATURE_COLUMNS if column not in bars.columns]
    if not missing:
        return bars.copy()
    features = compute_ict_features(bars)
    result = bars.copy()
    for column in FEATURE_COLUMNS:
        if column in features.columns and column not in result.columns:
            result[column] = features[column].to_numpy()
    return result


def attach_tick_bar_features(bars: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
    """Attach per-bar tick microstructure features from completed bar intervals.

    For each bar, only ticks in [bar_open_time, decision_time] are used. These
    are derived Level 1 tick features, not historical L2 order book depth.
    """
    if "time" not in bars.columns or "decision_time" not in bars.columns:
        raise ValueError("bars must contain time and decision_time columns")
    required_ticks = {"time_msc", "bid", "ask"}
    missing_ticks = required_ticks.difference(ticks.columns)
    if missing_ticks:
        raise ValueError(f"ticks missing required columns: {sorted(missing_ticks)}")

    result = bars.copy()
    ticks_sorted = ticks.sort_values("time_msc").reset_index(drop=True).copy()
    ticks_sorted["time_msc"] = pd.to_datetime(ticks_sorted["time_msc"], utc=True)
    tick_times = ticks_sorted["time_msc"].to_numpy(dtype="datetime64[ns]")
    bid = pd.to_numeric(ticks_sorted["bid"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(ticks_sorted["ask"], errors="coerce").to_numpy(dtype=float)
    mid = (bid + ask) / 2.0
    spread = ask - bid

    result["time"] = pd.to_datetime(result["time"], utc=True)
    result["decision_time"] = pd.to_datetime(result["decision_time"], utc=True)
    starts = np.searchsorted(tick_times, result["time"].to_numpy(dtype="datetime64[ns]"), side="left")
    ends = np.searchsorted(tick_times, result["decision_time"].to_numpy(dtype="datetime64[ns]"), side="right")

    values: dict[str, list[float]] = {column: [] for column in TICK_BAR_FEATURE_COLUMNS}
    for row_index, (start_idx, end_idx) in enumerate(zip(starts, ends, strict=True)):
        count = int(max(end_idx - start_idx, 0))
        bar_start = pd.Timestamp(result.loc[row_index, "time"])
        bar_end = pd.Timestamp(result.loc[row_index, "decision_time"])
        duration_seconds = max(float((bar_end - bar_start).total_seconds()), 1e-9)
        values["tick_count"].append(float(count))
        values["tick_rate_per_second"].append(float(count / duration_seconds))
        if count <= 0:
            for column in TICK_BAR_FEATURE_COLUMNS[2:]:
                values[column].append(0.0)
            continue

        mid_slice = mid[start_idx:end_idx]
        spread_slice = spread[start_idx:end_idx]
        mid_open = float(mid_slice[0])
        mid_close = float(mid_slice[-1])
        mid_high = float(np.max(mid_slice))
        mid_low = float(np.min(mid_slice))
        values["tick_spread_mean"].append(float(np.mean(spread_slice)))
        values["tick_spread_max"].append(float(np.max(spread_slice)))
        values["tick_mid_open"].append(mid_open)
        values["tick_mid_high"].append(mid_high)
        values["tick_mid_low"].append(mid_low)
        values["tick_mid_close"].append(mid_close)
        values["tick_mid_range"].append(mid_high - mid_low)
        values["tick_mid_return"].append((mid_close - mid_open) / max(abs(mid_open), 1e-12))

    for column, column_values in values.items():
        result[column] = column_values
    return result


def attach_decision_quotes(
    bars: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    max_quote_age_seconds: float = 5.0,
) -> pd.DataFrame:
    """Attach latest tick quote at or before each bar decision time.

    This precomputes execution quotes so repeated baseline/RL evaluations do not
    need to rescan millions of ticks for every strategy.
    """
    if "decision_time" not in bars.columns:
        raise ValueError("bars must include decision_time before attaching quotes")
    required_ticks = {"time_msc", "bid", "ask"}
    missing_ticks = required_ticks.difference(ticks.columns)
    if missing_ticks:
        raise ValueError(f"ticks missing required columns: {sorted(missing_ticks)}")

    bars_sorted = bars.sort_values("decision_time").reset_index(drop=True).copy()
    ticks_sorted = ticks.sort_values("time_msc").reset_index(drop=True).copy()
    bars_sorted["decision_time"] = pd.to_datetime(bars_sorted["decision_time"], utc=True)
    ticks_sorted["time_msc"] = pd.to_datetime(ticks_sorted["time_msc"], utc=True)

    quote_frame = ticks_sorted[["time_msc", "bid", "ask"]].rename(
        columns={"time_msc": "quote_time", "bid": "quote_bid", "ask": "quote_ask"}
    )
    merged = pd.merge_asof(
        bars_sorted,
        quote_frame,
        left_on="decision_time",
        right_on="quote_time",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["quote_time"] = pd.to_datetime(merged["quote_time"], utc=True)
    merged["quote_mid"] = (merged["quote_bid"] + merged["quote_ask"]) / 2.0
    merged["quote_spread"] = merged["quote_ask"] - merged["quote_bid"]
    merged["quote_age_seconds"] = (merged["decision_time"] - merged["quote_time"]).dt.total_seconds()
    merged["quote_is_fresh"] = (merged["quote_age_seconds"] >= 0.0) & (
        merged["quote_age_seconds"] <= max_quote_age_seconds
    )
    merged["quote_is_fresh"] = merged["quote_is_fresh"].fillna(False)
    return merged.sort_values("time").reset_index(drop=True)


def align_market_data(
    bars: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    timeframe: str | None = None,
    canonical_symbol: str | None = None,
    bar_source: str = "mt5_bars",
    tick_source: str = "exness_ticks",
    max_quote_age_seconds: float = 5.0,
    min_fresh_coverage: float = 0.95,
    include_ict_features: bool = True,
    include_tick_features: bool = True,
) -> AlignedDataset:
    """Align bar and tick frames into one canonical dataset.

    This resolves symbol aliases (e.g. XAUUSDm bars + XAUUSD archive ticks)
    into a canonical research symbol, adds bar decision_time, and computes tick
    freshness coverage for the exact aligned period.
    """
    if bars.empty:
        raise ValueError("bars cannot be empty")
    if ticks.empty:
        raise ValueError("ticks cannot be empty")

    bar_symbol = infer_single_symbol(bars)
    tick_symbol = infer_single_symbol(ticks)
    resolved_timeframe = (timeframe or infer_single_timeframe(bars)).upper()
    resolved_canonical = infer_canonical_symbol(bar_symbol, tick_symbol, canonical_symbol)

    aligned_bars = add_decision_time(bars, timeframe=resolved_timeframe).sort_values("time").reset_index(drop=True)
    aligned_ticks = ticks.copy().sort_values("time_msc").reset_index(drop=True)
    aligned_bars["time"] = pd.to_datetime(aligned_bars["time"], utc=True)
    aligned_bars["decision_time"] = pd.to_datetime(aligned_bars["decision_time"], utc=True)
    aligned_ticks["time"] = pd.to_datetime(aligned_ticks["time"], utc=True)
    aligned_ticks["time_msc"] = pd.to_datetime(aligned_ticks["time_msc"], utc=True)

    # Normalize symbols only after preserving original symbols in the manifest.
    aligned_bars["symbol"] = resolved_canonical
    aligned_ticks["symbol"] = resolved_canonical

    if include_ict_features:
        aligned_bars = attach_ict_features(aligned_bars)
    if include_tick_features:
        aligned_bars = attach_tick_bar_features(aligned_bars, aligned_ticks)

    coverage: TickBarCoverageDiagnostics = inspect_tick_bar_coverage(
        aligned_bars,
        aligned_ticks,
        timeframe=resolved_timeframe,
        max_quote_age_seconds=max_quote_age_seconds,
    )
    aligned_bars = attach_decision_quotes(
        aligned_bars,
        aligned_ticks,
        max_quote_age_seconds=max_quote_age_seconds,
    )
    fresh_ratio = compute_fresh_coverage_ratio(coverage)
    quality_status, quality_issues = dataset_quality_status(coverage, min_fresh_coverage=min_fresh_coverage)

    manifest = DatasetManifest(
        canonical_symbol=resolved_canonical,
        bar_symbol=bar_symbol,
        tick_symbol=tick_symbol,
        bar_source=bar_source,
        tick_source=tick_source,
        timeframe=resolved_timeframe,
        bars_rows=len(aligned_bars),
        ticks_rows=len(aligned_ticks),
        bars_start=str(aligned_bars["time"].min()),
        bars_end=str(aligned_bars["time"].max()),
        decision_start=str(aligned_bars["decision_time"].min()),
        decision_end=str(aligned_bars["decision_time"].max()),
        ticks_start=str(aligned_ticks["time_msc"].min()),
        ticks_end=str(aligned_ticks["time_msc"].max()),
        coverage=asdict(coverage),
        fresh_coverage_ratio=fresh_ratio,
        quality_status=quality_status,
        quality_issues=quality_issues,
    )
    return AlignedDataset(bars=aligned_bars, ticks=aligned_ticks, manifest=manifest)


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    frame.to_csv(path, index=False)
    return path


def save_aligned_dataset(
    dataset: AlignedDataset,
    output_dir: str | Path,
    *,
    source_bars_file: str | Path | None = None,
    source_ticks_file: str | Path | None = None,
    copy_ticks: bool = False,
) -> DatasetManifest:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bars_path = _write_frame(dataset.bars, output / "bars.parquet")
    if copy_ticks or source_ticks_file is None:
        ticks_path = _write_frame(dataset.ticks, output / "ticks.parquet")
    else:
        ticks_path = Path(source_ticks_file)
    manifest_path = output / "manifest.json"
    files = {
        "bars": str(bars_path),
        "ticks": str(ticks_path),
        "manifest": str(manifest_path),
    }
    source_files = {
        "bars": str(source_bars_file) if source_bars_file is not None else "",
        "ticks": str(source_ticks_file) if source_ticks_file is not None else "",
    }
    manifest = DatasetManifest(**{**asdict(dataset.manifest), "files": files, "source_files": source_files})
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: str | Path) -> DatasetManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetManifest(**data)


def render_manifest(manifest: DatasetManifest, *, console: Console | None = None) -> None:
    target = console or Console()
    table = Table(title="Aligned Dataset Manifest")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for key, value in asdict(manifest).items():
        if key == "coverage":
            for cov_key, cov_value in value.items():
                table.add_row(f"coverage.{cov_key}", str(cov_value))
        elif key == "files":
            for file_key, file_value in value.items():
                table.add_row(f"files.{file_key}", str(file_value))
        elif key == "source_files":
            for file_key, file_value in value.items():
                table.add_row(f"source_files.{file_key}", str(file_value))
        elif isinstance(value, list):
            table.add_row(key, ", ".join(str(item) for item in value))
        else:
            table.add_row(key, str(value))
    target.print(table)
