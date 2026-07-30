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
def run_backtest(
    bars_file: str = typer.Option(..., help="Canonical bars file (.csv or .parquet)"),
    strategy: str = typer.Option("no-trade", help="no-trade, buy-and-hold, ma-cross, or ict-bias"),
    symbol: str | None = typer.Option(None, help="Symbol override if the file contains multiple symbols"),
    volume: float = typer.Option(0.1, help="Order volume used by baseline strategies"),
    initial_balance: float = typer.Option(100_000.0, help="Initial account balance"),
    point_size: float = typer.Option(0.01, help="Instrument point size"),
    point_value: float = typer.Option(1.0, help="PnL value per price unit and volume"),
    default_spread_points: float = typer.Option(20.0, help="Fallback spread in points when bars have no spread"),
    slippage_points: float = typer.Option(0.0, help="Adverse slippage in points"),
    commission_per_volume: float = typer.Option(0.0, help="Commission per traded volume unit"),
    fast_window: int = typer.Option(5, help="Fast MA window for ma-cross"),
    slow_window: int = typer.Option(20, help="Slow MA window for ma-cross"),
) -> None:
    """Run a baseline backtest from a canonical bars file."""
    from slytrade.backtest.engine import BacktestConfig
    from slytrade.backtest.reporting import (
        VALID_STRATEGIES,
        load_bars_file,
        render_backtest_report,
        run_backtest_from_bars,
    )

    if strategy not in VALID_STRATEGIES:
        raise typer.BadParameter(f"strategy must be one of: {', '.join(VALID_STRATEGIES)}")
    bars = load_bars_file(Path(bars_file))
    result = run_backtest_from_bars(
        bars,
        strategy_name=strategy,
        symbol=symbol,
        volume=volume,
        fast_window=fast_window,
        slow_window=slow_window,
        config=BacktestConfig(
            initial_balance=initial_balance,
            default_spread_points=default_spread_points,
            point_size=point_size,
            point_value=point_value,
            slippage_points=slippage_points,
            commission_per_volume=commission_per_volume,
        ),
    )
    render_backtest_report(result, strategy_name=strategy, console=console)


@app.command()
def compare_baselines(
    bars_file: str = typer.Option(..., help="Canonical bars file (.csv or .parquet)"),
    symbol: str | None = typer.Option(None, help="Symbol override if the file contains multiple symbols"),
    volume: float = typer.Option(0.1, help="Order volume used by baseline strategies"),
    initial_balance: float = typer.Option(100_000.0, help="Initial account balance"),
    point_size: float = typer.Option(0.01, help="Instrument point size"),
    point_value: float = typer.Option(1.0, help="PnL value per price unit and volume"),
    default_spread_points: float = typer.Option(20.0, help="Fallback spread in points when bars have no spread"),
    slippage_points: float = typer.Option(0.0, help="Adverse slippage in points"),
    commission_per_volume: float = typer.Option(0.0, help="Commission per traded volume unit"),
    fast_window: int = typer.Option(5, help="Fast MA window for ma-cross"),
    slow_window: int = typer.Option(20, help="Slow MA window for ma-cross"),
    output_csv: str | None = typer.Option(None, help="Optional path to save comparison as CSV"),
) -> None:
    """Run all baseline strategies and print a comparison table."""
    from slytrade.backtest.engine import BacktestConfig
    from slytrade.backtest.reporting import (
        compare_baselines_from_bars,
        comparison_as_frame,
        load_bars_file,
        render_baseline_comparison,
    )

    bars = load_bars_file(Path(bars_file))
    rows = compare_baselines_from_bars(
        bars,
        symbol=symbol,
        volume=volume,
        fast_window=fast_window,
        slow_window=slow_window,
        config=BacktestConfig(
            initial_balance=initial_balance,
            default_spread_points=default_spread_points,
            point_size=point_size,
            point_value=point_value,
            slippage_points=slippage_points,
            commission_per_volume=commission_per_volume,
        ),
    )
    render_baseline_comparison(rows, console=console)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_as_frame(rows).to_csv(output_path, index=False)
        console.print(f"Saved comparison CSV to {output_path}")


@app.command()
def generate_sample_bars(
    output_file: str = typer.Option(..., help="Output .csv or .parquet file"),
    symbol: str = typer.Option("XAUUSD", help="Sample symbol"),
    timeframe: str = typer.Option("M1", help="Sample timeframe"),
    start: str = typer.Option("2026-01-01", help="UTC start date/datetime"),
    periods: int = typer.Option(500, help="Number of bars to generate"),
    seed: int = typer.Option(42, help="Deterministic random seed"),
    start_price: float = typer.Option(2400.0, help="Starting price"),
) -> None:
    """Generate deterministic sample bars for demos and tests."""
    from slytrade.data.sample_generator import generate_sample_bars as generate_bars
    from slytrade.data.sample_generator import write_sample_frame

    frame = generate_bars(
        symbol=symbol,
        timeframe=timeframe,
        start=parse_utc_datetime(start),
        periods=periods,
        seed=seed,
        start_price=start_price,
    )
    path = write_sample_frame(frame, output_file)
    console.print(f"Generated {len(frame)} sample bars at {path}")


@app.command()
def generate_sample_ticks(
    output_file: str = typer.Option(..., help="Output .csv or .parquet file"),
    symbol: str = typer.Option("XAUUSD", help="Sample symbol"),
    start: str = typer.Option("2026-01-01", help="UTC start date/datetime"),
    periods: int = typer.Option(2_000, help="Number of ticks to generate"),
    seed: int = typer.Option(42, help="Deterministic random seed"),
    start_price: float = typer.Option(2400.0, help="Starting mid price"),
) -> None:
    """Generate deterministic sample ticks for demos and tests."""
    from slytrade.data.sample_generator import generate_sample_ticks as generate_ticks
    from slytrade.data.sample_generator import write_sample_frame

    frame = generate_ticks(
        symbol=symbol,
        start=parse_utc_datetime(start),
        periods=periods,
        seed=seed,
        start_price=start_price,
    )
    path = write_sample_frame(frame, output_file)
    console.print(f"Generated {len(frame)} sample ticks at {path}")


@app.command()
def run_tick_backtest(
    bars_file: str = typer.Option(..., help="Canonical bars file (.csv or .parquet)"),
    ticks_file: str = typer.Option(..., help="Canonical ticks file (.csv or .parquet)"),
    strategy: str = typer.Option("no-trade", help="no-trade, buy-and-hold, ma-cross, or ict-bias"),
    symbol: str | None = typer.Option(None, help="Symbol override if the bars file contains multiple symbols"),
    volume: float = typer.Option(0.1, help="Order volume used by baseline strategies"),
    initial_balance: float = typer.Option(100_000.0, help="Initial account balance"),
    point_size: float = typer.Option(0.01, help="Instrument point size"),
    point_value: float = typer.Option(1.0, help="PnL value per price unit and volume"),
    default_spread_points: float = typer.Option(20.0, help="Fallback spread in points when no tick quote exists"),
    slippage_points: float = typer.Option(0.0, help="Adverse slippage in points"),
    commission_per_volume: float = typer.Option(0.0, help="Commission per traded volume unit"),
    fast_window: int = typer.Option(5, help="Fast MA window for ma-cross"),
    slow_window: int = typer.Option(20, help="Slow MA window for ma-cross"),
) -> None:
    """Run a baseline backtest where bar signals execute on tick bid/ask quotes."""
    from slytrade.backtest.engine import BacktestConfig
    from slytrade.backtest.reporting import (
        VALID_STRATEGIES,
        load_bars_file,
        load_ticks_file,
        render_backtest_report,
        run_tick_backtest_from_frames,
    )

    if strategy not in VALID_STRATEGIES:
        raise typer.BadParameter(f"strategy must be one of: {', '.join(VALID_STRATEGIES)}")
    bars = load_bars_file(Path(bars_file))
    ticks = load_ticks_file(Path(ticks_file))
    result = run_tick_backtest_from_frames(
        bars,
        ticks,
        strategy_name=strategy,
        symbol=symbol,
        volume=volume,
        fast_window=fast_window,
        slow_window=slow_window,
        config=BacktestConfig(
            initial_balance=initial_balance,
            default_spread_points=default_spread_points,
            point_size=point_size,
            point_value=point_value,
            slippage_points=slippage_points,
            commission_per_volume=commission_per_volume,
        ),
    )
    render_backtest_report(result, strategy_name=f"tick-{strategy}", console=console)


@app.command()
def live() -> None:
    """Live trading placeholder."""
    console.print("[bold red]Live trading is disabled at bootstrap stage.[/bold red]")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
