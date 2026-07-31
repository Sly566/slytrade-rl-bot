from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from rich.console import Console
from rich.table import Table

from slytrade.data.timeframes import add_decision_time


@dataclass(frozen=True)
class BarDiagnostics:
    rows: int
    symbol: str
    timeframe: str
    start_time: str
    end_time: str
    start_decision_time: str
    end_decision_time: str
    duplicate_bars: int
    invalid_ohlc_rows: int
    missing_time_rows: int


@dataclass(frozen=True)
class TickDiagnostics:
    rows: int
    symbol: str
    start_time: str
    end_time: str
    duplicate_ticks: int
    bad_price_rows: int
    crossed_spread_rows: int
    spread_min: float
    spread_mean: float
    spread_max: float


@dataclass(frozen=True)
class TickBarCoverageDiagnostics:
    bars: int
    bars_with_tick_before_decision: int
    bars_missing_tick_before_decision: int
    first_missing_decision_time: str | None


def _single_value(frame: pd.DataFrame, column: str, fallback: str = "unknown") -> str:
    if column not in frame.columns or frame.empty:
        return fallback
    values = sorted(str(value) for value in frame[column].dropna().unique())
    if len(values) == 1:
        return values[0]
    if not values:
        return fallback
    return "multiple"


def inspect_bars(bars: pd.DataFrame, *, timeframe: str | None = None) -> BarDiagnostics:
    if bars.empty:
        return BarDiagnostics(0, "unknown", timeframe or "unknown", "", "", "", "", 0, 0, 0)
    required = {"time", "symbol", "timeframe", "open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    aligned = add_decision_time(bars, timeframe=timeframe)
    duplicate_bars = int(aligned.duplicated(subset=["time", "symbol", "timeframe"]).sum())
    missing_time_rows = int(aligned["time"].isna().sum())
    invalid_ohlc_rows = int(
        (
            (aligned["high"] < aligned["low"])
            | (aligned["high"] < aligned[["open", "close"]].max(axis=1))
            | (aligned["low"] > aligned[["open", "close"]].min(axis=1))
        ).sum()
    )
    return BarDiagnostics(
        rows=len(aligned),
        symbol=_single_value(aligned, "symbol"),
        timeframe=timeframe or _single_value(aligned, "timeframe"),
        start_time=str(aligned["time"].min()),
        end_time=str(aligned["time"].max()),
        start_decision_time=str(aligned["decision_time"].min()),
        end_decision_time=str(aligned["decision_time"].max()),
        duplicate_bars=duplicate_bars,
        invalid_ohlc_rows=invalid_ohlc_rows,
        missing_time_rows=missing_time_rows,
    )


def inspect_ticks(ticks: pd.DataFrame) -> TickDiagnostics:
    if ticks.empty:
        return TickDiagnostics(0, "unknown", "", "", 0, 0, 0, 0.0, 0.0, 0.0)
    required = {"time_msc", "symbol", "bid", "ask"}
    missing = required.difference(ticks.columns)
    if missing:
        raise ValueError(f"ticks missing required columns: {sorted(missing)}")

    clean = ticks.copy()
    clean["time_msc"] = pd.to_datetime(clean["time_msc"], utc=True)
    if "last" not in clean.columns:
        clean["last"] = 0.0
    spread = clean["ask"] - clean["bid"]
    duplicate_ticks = int(clean.duplicated(subset=["time_msc", "bid", "ask", "last"]).sum())
    bad_price_rows = int(((clean["bid"] <= 0) | (clean["ask"] <= 0)).sum())
    crossed_spread_rows = int((clean["ask"] < clean["bid"]).sum())
    return TickDiagnostics(
        rows=len(clean),
        symbol=_single_value(clean, "symbol"),
        start_time=str(clean["time_msc"].min()),
        end_time=str(clean["time_msc"].max()),
        duplicate_ticks=duplicate_ticks,
        bad_price_rows=bad_price_rows,
        crossed_spread_rows=crossed_spread_rows,
        spread_min=float(spread.min()),
        spread_mean=float(spread.mean()),
        spread_max=float(spread.max()),
    )


def inspect_tick_bar_coverage(bars: pd.DataFrame, ticks: pd.DataFrame, *, timeframe: str | None = None) -> TickBarCoverageDiagnostics:
    if bars.empty:
        return TickBarCoverageDiagnostics(0, 0, 0, None)
    if ticks.empty:
        aligned = add_decision_time(bars, timeframe=timeframe)
        return TickBarCoverageDiagnostics(len(aligned), 0, len(aligned), str(aligned["decision_time"].min()))

    aligned = add_decision_time(bars, timeframe=timeframe).sort_values("decision_time").reset_index(drop=True)
    ticks_sorted = ticks.copy()
    ticks_sorted["time_msc"] = pd.to_datetime(ticks_sorted["time_msc"], utc=True)
    ticks_sorted = ticks_sorted.sort_values("time_msc").reset_index(drop=True)

    tick_index = 0
    seen = 0
    missing_times: list[str] = []
    for _, bar in aligned.iterrows():
        decision_time = pd.Timestamp(bar["decision_time"])
        while tick_index < len(ticks_sorted) and pd.Timestamp(ticks_sorted.loc[tick_index, "time_msc"]) <= decision_time:
            tick_index += 1
        if tick_index > 0:
            seen += 1
        else:
            missing_times.append(str(decision_time))
    missing = len(aligned) - seen
    return TickBarCoverageDiagnostics(
        bars=len(aligned),
        bars_with_tick_before_decision=seen,
        bars_missing_tick_before_decision=missing,
        first_missing_decision_time=missing_times[0] if missing_times else None,
    )


def render_data_diagnostics(
    *,
    bars: BarDiagnostics | None = None,
    ticks: TickDiagnostics | None = None,
    coverage: TickBarCoverageDiagnostics | None = None,
    console: Console | None = None,
) -> None:
    target = console or Console()
    table = Table(title="Data Diagnostics")
    table.add_column("Section")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    if bars is not None:
        for key, value in asdict(bars).items():
            table.add_row("bars", key, str(value))
    if ticks is not None:
        for key, value in asdict(ticks).items():
            table.add_row("ticks", key, str(value))
    if coverage is not None:
        for key, value in asdict(coverage).items():
            table.add_row("coverage", key, str(value))
    target.print(table)
