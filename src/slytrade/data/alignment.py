from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from slytrade.data.diagnostics import TickBarCoverageDiagnostics, inspect_tick_bar_coverage
from slytrade.data.exness_archive import normalize_exness_symbol
from slytrade.data.timeframes import add_decision_time

QUOTE_COLUMNS = [
    "quote_time",
    "quote_bid",
    "quote_ask",
    "quote_mid",
    "quote_spread",
    "quote_age_seconds",
    "quote_is_fresh",
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


def save_aligned_dataset(dataset: AlignedDataset, output_dir: str | Path) -> DatasetManifest:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bars_path = _write_frame(dataset.bars, output / "bars.parquet")
    ticks_path = _write_frame(dataset.ticks, output / "ticks.parquet")
    manifest_path = output / "manifest.json"
    files = {
        "bars": str(bars_path),
        "ticks": str(ticks_path),
        "manifest": str(manifest_path),
    }
    manifest = DatasetManifest(**{**asdict(dataset.manifest), "files": files})
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
        else:
            table.add_row(key, str(value))
    target.print(table)
