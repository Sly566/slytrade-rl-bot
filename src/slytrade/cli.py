from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from slytrade.data.time import parse_utc_datetime

app = typer.Typer(help="SlyTrade RL Bot CLI")
console = Console()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_mt5() -> Any:
    """Load an MT5-compatible Python module/object lazily.

    In Windows-native environments this usually imports `MetaTrader5`.
    In Linux bridge environments this may use `mt5linux`.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]

        return mt5
    except ImportError:
        try:
            from mt5linux import MetaTrader5 as MT5Linux  # type: ignore[import-not-found]

            return MT5Linux()
        except ImportError as exc:
            raise RuntimeError(
                "No MT5 Python integration found. Install the `mt5` optional dependencies and start your MT5 bridge."
            ) from exc


def initialize_mt5(mt5: Any) -> None:
    if hasattr(mt5, "initialize"):
        ok = mt5.initialize()
        if ok is False:
            raise RuntimeError("MT5 initialize() returned False")


def shutdown_mt5(mt5: Any) -> None:
    if hasattr(mt5, "shutdown"):
        mt5.shutdown()


@app.command()
def doctor() -> None:
    """Check project health and dependencies."""
    table = Table(title="SlyTrade RL Bot Doctor")
    table.add_column("Check")
    table.add_column("Status")

    required = ["numpy", "pandas", "yaml", "pydantic", "typer", "rich"]
    optional = [
        "pyarrow",
        "polars",
        "torch",
        "gymnasium",
        "stable_baselines3",
        "optuna",
        "mlflow",
        "mt5linux",
    ]

    for mod in required:
        table.add_row(f"required:{mod}", "OK" if module_available(mod) else "MISSING")

    for mod in optional:
        table.add_row(f"optional:{mod}", "OK" if module_available(mod) else "MISSING")

    for path in [
        "configs/assets.yaml",
        "configs/broker.yaml",
        "configs/data.yaml",
        "configs/risk.yaml",
        "configs/training.yaml",
    ]:
        table.add_row(f"file:{path}", "OK" if Path(path).exists() else "MISSING")

    console.print(table)


@app.command()
def info() -> None:
    """Print project information."""
    console.print("[bold green]SlyTrade RL Bot[/bold green]")
    console.print("Production-grade MT5 tick-and-bar based ICT/SMC RL trading system.")
    console.print("Live trading is disabled by default.")


@app.command()
def collect_ticks(
    symbol: str = typer.Option(..., help="Resolved MT5 symbol, e.g. XAUUSD or XAUUSDm"),
    start: str = typer.Option(..., help="UTC start date/datetime, e.g. 2026-01-01"),
    end: str = typer.Option(..., help="UTC end date/datetime, e.g. 2026-01-02"),
    chunk_size: str = typer.Option("day", help="day, week, or month"),
    output_dir: str = typer.Option("data/raw", help="Raw data root directory"),
) -> None:
    """Collect MT5 historical ticks into partitioned storage."""
    from slytrade.data.mt5_collectors import MT5TickCollector
    from slytrade.data.storage import MarketDataStorage

    mt5 = load_mt5()
    initialize_mt5(mt5)
    try:
        collector = MT5TickCollector(mt5, MarketDataStorage(Path(output_dir)))
        result = collector.collect(symbol, parse_utc_datetime(start), parse_utc_datetime(end), chunk_size=chunk_size)  # type: ignore[arg-type]
        console.print(f"Collected {result.rows} tick rows into {result.file_count} files")
        for file_result in result.files:
            console.print(f"- {file_result.path} ({file_result.rows} rows, {file_result.format})")
    finally:
        shutdown_mt5(mt5)


@app.command()
def collect_bars(
    symbol: str = typer.Option(..., help="Resolved MT5 symbol, e.g. XAUUSD or XAUUSDm"),
    timeframe: str = typer.Option(..., help="Timeframe, e.g. M1, M5, H1"),
    start: str = typer.Option(..., help="UTC start date/datetime, e.g. 2026-01-01"),
    end: str = typer.Option(..., help="UTC end date/datetime, e.g. 2026-01-02"),
    chunk_size: str = typer.Option("month", help="day, week, or month"),
    output_dir: str = typer.Option("data/raw", help="Raw data root directory"),
) -> None:
    """Collect MT5 historical bars/rates into partitioned storage."""
    from slytrade.data.mt5_collectors import MT5BarCollector
    from slytrade.data.storage import MarketDataStorage

    mt5 = load_mt5()
    initialize_mt5(mt5)
    try:
        collector = MT5BarCollector(mt5, MarketDataStorage(Path(output_dir)))
        result = collector.collect(
            symbol,
            timeframe,
            parse_utc_datetime(start),
            parse_utc_datetime(end),
            chunk_size=chunk_size,  # type: ignore[arg-type]
        )
        console.print(f"Collected {result.rows} bar rows into {result.file_count} files")
        for file_result in result.files:
            console.print(f"- {file_result.path} ({file_result.rows} rows, {file_result.format})")
    finally:
        shutdown_mt5(mt5)


@app.command()
def live() -> None:
    """Live trading placeholder."""
    console.print("[bold red]Live trading is disabled at bootstrap stage.[/bold red]")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
