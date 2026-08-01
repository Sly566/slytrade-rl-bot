from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from slytrade.data.time import date_range_from_lookback, parse_utc_datetime

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
        except Exception as exc:
            message = str(exc)
            if "invalid message type" in message.lower():
                raise RuntimeError(
                    "mt5linux connected to an incompatible or stale RPyC server. "
                    "Restart the Wine mt5linux server and make sure Linux and Wine Python use compatible mt5linux/rpyc versions."
                ) from exc
            raise RuntimeError(f"Could not connect to mt5linux bridge: {exc}") from exc


def initialize_mt5(mt5: Any) -> None:
    if hasattr(mt5, "initialize"):
        ok = mt5.initialize()
        if ok is False:
            raise RuntimeError("MT5 initialize() returned False")


def shutdown_mt5(mt5: Any) -> None:
    if hasattr(mt5, "shutdown"):
        mt5.shutdown()


def build_backtest_config_from_cli(
    *,
    initial_balance: float,
    point_size: float,
    point_value: float,
    default_spread_points: float = 20.0,
    slippage_points: float = 0.0,
    commission_per_volume: float = 0.0,
    max_quote_age_seconds: float = 5.0,
    allow_bar_quote_fallback: bool = True,
    symbol_spec_file: str | None = None,
):
    from slytrade.backtest.engine import BacktestConfig

    if symbol_spec_file is not None:
        from slytrade.brokers.specs import load_symbol_spec, spec_to_backtest_pricing

        spec = load_symbol_spec(symbol_spec_file)
        pricing = spec_to_backtest_pricing(spec)
        point_size = pricing.point_size
        point_value = pricing.point_value
        console.print(
            f"Using symbol spec {spec.name}: point_size={point_size}, "
            f"point_value={point_value:.6f}, volume_step={pricing.volume_step}"
        )
    return BacktestConfig(
        initial_balance=initial_balance,
        default_spread_points=default_spread_points,
        point_size=point_size,
        point_value=point_value,
        slippage_points=slippage_points,
        commission_per_volume=commission_per_volume,
        max_quote_age_seconds=max_quote_age_seconds,
        allow_bar_quote_fallback=allow_bar_quote_fallback,
    )


def print_collection_result(result: Any, original_symbol: str) -> None:
    if result.symbol != original_symbol:
        console.print(f"Resolved symbol: {original_symbol} -> {result.symbol}")
    console.print(
        f"Collected {result.rows} {result.dataset} rows into {result.file_count} files "
        f"({result.chunks_attempted} chunks attempted, {result.empty_chunks} empty chunks)"
    )
    if result.empty_chunks:
        console.print(
            "[yellow]Warning:[/yellow] some chunks returned no data. "
            "For ticks this may indicate broker/terminal tick-history limits."
        )
    for file_result in result.files:
        console.print(f"- {file_result.path} ({file_result.rows} rows, {file_result.format})")


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
    symbol: str = typer.Option(..., help="Base or resolved MT5 symbol, e.g. XAUUSD or XAUUSDm"),
    start: str = typer.Option(..., help="UTC start date/datetime, e.g. 2026-01-01"),
    end: str = typer.Option(..., help="UTC end date/datetime, e.g. 2026-01-02"),
    chunk_size: str = typer.Option("day", help="day, week, or month"),
    output_dir: str = typer.Option("data/raw", help="Raw data root directory"),
    resolve: bool = typer.Option(True, "--resolve/--no-resolve", help="Resolve base symbol to broker-specific MT5 symbol"),
) -> None:
    """Collect MT5 historical ticks into partitioned storage."""
    from slytrade.data.mt5_collectors import MT5TickCollector
    from slytrade.data.storage import MarketDataStorage

    mt5 = load_mt5()
    initialize_mt5(mt5)
    try:
        collector = MT5TickCollector(mt5, MarketDataStorage(Path(output_dir)))
        result = collector.collect(symbol, parse_utc_datetime(start), parse_utc_datetime(end), chunk_size=chunk_size, resolve=resolve)  # type: ignore[arg-type]
        print_collection_result(result, symbol)
    finally:
        shutdown_mt5(mt5)


@app.command()
def collect_bars(
    symbol: str = typer.Option(..., help="Base or resolved MT5 symbol, e.g. XAUUSD or XAUUSDm"),
    timeframe: str = typer.Option(..., help="Timeframe, e.g. M1, M5, H1"),
    start: str = typer.Option(..., help="UTC start date/datetime, e.g. 2026-01-01"),
    end: str = typer.Option(..., help="UTC end date/datetime, e.g. 2026-01-02"),
    chunk_size: str = typer.Option("month", help="day, week, or month"),
    output_dir: str = typer.Option("data/raw", help="Raw data root directory"),
    resolve: bool = typer.Option(True, "--resolve/--no-resolve", help="Resolve base symbol to broker-specific MT5 symbol"),
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
            resolve=resolve,
        )
        print_collection_result(result, symbol)
    finally:
        shutdown_mt5(mt5)


@app.command()
def collect_recent_bars(
    symbol: str = typer.Option(..., help="Base or resolved MT5 symbol, e.g. XAUUSD or XAUUSDm"),
    timeframe: str = typer.Option(..., help="Timeframe, e.g. M1, M5, H1"),
    lookback: str = typer.Option("1y", help="Lookback duration, e.g. 1d, 1w, 1m, 1y, 2y"),
    end: str | None = typer.Option(None, help="UTC end date/datetime. Defaults to current UTC time."),
    chunk_size: str = typer.Option("month", help="day, week, or month"),
    output_dir: str = typer.Option("data/raw", help="Raw data root directory"),
    resolve: bool = typer.Option(True, "--resolve/--no-resolve", help="Resolve base symbol to broker-specific MT5 symbol"),
) -> None:
    """Collect recent MT5 bars using a relative lookback from now or --end."""
    from slytrade.data.mt5_collectors import MT5BarCollector
    from slytrade.data.storage import MarketDataStorage

    end_dt = parse_utc_datetime(end) if end is not None else None
    start_dt, resolved_end = date_range_from_lookback(lookback, end=end_dt)
    console.print(f"Collecting bars from {start_dt.isoformat()} to {resolved_end.isoformat()} (lookback={lookback})")
    mt5 = load_mt5()
    initialize_mt5(mt5)
    try:
        collector = MT5BarCollector(mt5, MarketDataStorage(Path(output_dir)))
        result = collector.collect(
            symbol,
            timeframe,
            start_dt,
            resolved_end,
            chunk_size=chunk_size,  # type: ignore[arg-type]
            resolve=resolve,
        )
        print_collection_result(result, symbol)
    finally:
        shutdown_mt5(mt5)


@app.command()
def collect_recent_ticks(
    symbol: str = typer.Option(..., help="Base or resolved MT5 symbol, e.g. XAUUSD or XAUUSDm"),
    lookback: str = typer.Option("1m", help="Lookback duration, e.g. 1d, 1w, 1m, 1y, 2y"),
    end: str | None = typer.Option(None, help="UTC end date/datetime. Defaults to current UTC time."),
    chunk_size: str = typer.Option("day", help="day, week, or month"),
    output_dir: str = typer.Option("data/raw", help="Raw data root directory"),
    resolve: bool = typer.Option(True, "--resolve/--no-resolve", help="Resolve base symbol to broker-specific MT5 symbol"),
) -> None:
    """Collect recent MT5 ticks using a relative lookback from now or --end."""
    from slytrade.data.mt5_collectors import MT5TickCollector
    from slytrade.data.storage import MarketDataStorage

    end_dt = parse_utc_datetime(end) if end is not None else None
    start_dt, resolved_end = date_range_from_lookback(lookback, end=end_dt)
    console.print(f"Collecting ticks from {start_dt.isoformat()} to {resolved_end.isoformat()} (lookback={lookback})")
    mt5 = load_mt5()
    initialize_mt5(mt5)
    try:
        collector = MT5TickCollector(mt5, MarketDataStorage(Path(output_dir)))
        result = collector.collect(
            symbol,
            start_dt,
            resolved_end,
            chunk_size=chunk_size,  # type: ignore[arg-type]
            resolve=resolve,
        )
        print_collection_result(result, symbol)
    finally:
        shutdown_mt5(mt5)


def print_exness_archive_result(result: Any) -> None:
    console.print(
        f"Collected {result.rows} Exness archive tick rows into {result.file_count} files "
        f"({result.months_attempted} months attempted, {result.empty_months} empty, {result.failed_months} failed)"
    )
    if result.errors:
        console.print("[yellow]Warnings/errors:[/yellow]")
        for error in result.errors[:20]:
            console.print(f"- {error}")
        if len(result.errors) > 20:
            console.print(f"... {len(result.errors) - 20} more")
    for file_result in result.files:
        console.print(f"- {file_result.path} ({file_result.rows} rows, {file_result.format}, {file_result.period})")


@app.command()
def collect_exness_ticks(
    symbol: str = typer.Option(..., help="Exness archive base symbol, e.g. XAUUSD"),
    start: str = typer.Option(..., help="UTC start date/datetime, e.g. 2025-07-01"),
    end: str = typer.Option(..., help="UTC end date/datetime, e.g. 2026-07-01"),
    output_dir: str = typer.Option("data/raw", help="Raw data root directory"),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue when an archive month is unavailable",
    ),
) -> None:
    """Collect historical ticks directly from the Exness public archive."""
    from slytrade.data.exness_archive import ExnessArchiveDownloader

    downloader = ExnessArchiveDownloader(output_dir)
    result = downloader.collect(
        symbol,
        parse_utc_datetime(start),
        parse_utc_datetime(end),
        continue_on_error=continue_on_error,
    )
    print_exness_archive_result(result)


@app.command()
def collect_recent_exness_ticks(
    symbol: str = typer.Option(..., help="Exness archive base symbol, e.g. XAUUSD"),
    lookback: str = typer.Option("1y", help="Lookback duration, e.g. 1m, 6m, 1y, 2y"),
    end: str | None = typer.Option(None, help="UTC end date/datetime. Defaults to current UTC time."),
    output_dir: str = typer.Option("data/raw", help="Raw data root directory"),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue when an archive month is unavailable",
    ),
) -> None:
    """Collect Exness archive ticks using a relative lookback from now or --end."""
    from slytrade.data.exness_archive import ExnessArchiveDownloader

    end_dt = parse_utc_datetime(end) if end is not None else None
    start_dt, resolved_end = date_range_from_lookback(lookback, end=end_dt)
    console.print(
        f"Collecting Exness archive ticks from {start_dt.isoformat()} to {resolved_end.isoformat()} "
        f"(lookback={lookback})"
    )
    downloader = ExnessArchiveDownloader(output_dir)
    result = downloader.collect(symbol, start_dt, resolved_end, continue_on_error=continue_on_error)
    print_exness_archive_result(result)


@app.command()
def collect_symbol_spec(
    symbol: str = typer.Option(..., help="Base or resolved MT5 symbol, e.g. XAUUSD"),
    output_file: str | None = typer.Option(None, help="Output JSON file. Defaults under data/raw/symbol_specs."),
    resolve: bool = typer.Option(True, "--resolve/--no-resolve", help="Resolve base symbol to broker-specific MT5 symbol"),
) -> None:
    """Collect broker symbol specs from MT5 for realistic PnL and sizing."""
    from slytrade.brokers.specs import save_symbol_spec, symbol_spec_from_mt5_info
    from slytrade.brokers.symbols import resolve_symbol

    mt5 = load_mt5()
    initialize_mt5(mt5)
    try:
        actual_symbol = resolve_symbol(mt5, symbol).resolved if resolve else symbol
        info = mt5.symbol_info(actual_symbol)
        spec = symbol_spec_from_mt5_info(info)
        path = (
            Path(output_file)
            if output_file is not None
            else Path("data/raw/symbol_specs") / f"{spec.name}.json"
        )
        save_symbol_spec(spec, path)
        if actual_symbol != symbol:
            console.print(f"Resolved symbol: {symbol} -> {actual_symbol}")
        console.print(f"Saved symbol spec to {path}")
        console.print(
            f"point_size={spec.trade_tick_size}, point_value={spec.point_value_per_price_unit:.6f}, "
            f"volume_min={spec.volume_min}, volume_step={spec.volume_step}"
        )
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
    symbol_spec_file: str | None = typer.Option(None, help="Optional symbol spec JSON to set point size/value"),
    default_spread_points: float = typer.Option(20.0, help="Fallback spread in points when bars have no spread"),
    slippage_points: float = typer.Option(0.0, help="Adverse slippage in points"),
    commission_per_volume: float = typer.Option(0.0, help="Commission per traded volume unit"),
    fast_window: int = typer.Option(5, help="Fast MA window for ma-cross"),
    slow_window: int = typer.Option(20, help="Slow MA window for ma-cross"),
) -> None:
    """Run a baseline backtest from a canonical bars file."""
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
        config=build_backtest_config_from_cli(
            initial_balance=initial_balance,
            default_spread_points=default_spread_points,
            point_size=point_size,
            point_value=point_value,
            slippage_points=slippage_points,
            commission_per_volume=commission_per_volume,
            symbol_spec_file=symbol_spec_file,
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
    symbol_spec_file: str | None = typer.Option(None, help="Optional symbol spec JSON to set point size/value"),
    default_spread_points: float = typer.Option(20.0, help="Fallback spread in points when bars have no spread"),
    slippage_points: float = typer.Option(0.0, help="Adverse slippage in points"),
    commission_per_volume: float = typer.Option(0.0, help="Commission per traded volume unit"),
    fast_window: int = typer.Option(5, help="Fast MA window for ma-cross"),
    slow_window: int = typer.Option(20, help="Slow MA window for ma-cross"),
    output_csv: str | None = typer.Option(None, help="Optional path to save comparison as CSV"),
) -> None:
    """Run all baseline strategies and print a comparison table."""
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
        config=build_backtest_config_from_cli(
            initial_balance=initial_balance,
            default_spread_points=default_spread_points,
            point_size=point_size,
            point_value=point_value,
            slippage_points=slippage_points,
            commission_per_volume=commission_per_volume,
            symbol_spec_file=symbol_spec_file,
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
def align_dataset(
    bars_file: str = typer.Option(..., help="Canonical bars file (.csv or .parquet)"),
    ticks_file: str = typer.Option(..., help="Canonical ticks file (.csv or .parquet)"),
    output_dir: str = typer.Option(..., help="Directory for aligned bars/ticks/manifest"),
    timeframe: str = typer.Option("M1", help="Bar timeframe, e.g. M1"),
    canonical_symbol: str | None = typer.Option(None, help="Canonical research symbol, e.g. XAUUSD"),
    bar_source: str = typer.Option("mt5_bars", help="Bar source label"),
    tick_source: str = typer.Option("exness_ticks", help="Tick source label"),
    max_quote_age_seconds: float = typer.Option(5.0, help="Fresh tick coverage threshold"),
    min_fresh_coverage: float = typer.Option(0.95, help="Minimum expected fresh tick coverage ratio"),
    include_ict_features: bool = typer.Option(True, "--features/--no-features", help="Compute causal ICT/SMC features"),
    include_tick_features: bool = typer.Option(True, "--tick-features/--no-tick-features", help="Compute per-bar tick microstructure features"),
    require_fresh_quotes: bool = typer.Option(True, "--fresh-only/--keep-stale", help="Drop bars without fresh decision quotes"),
    mtf: bool = typer.Option(False, "--mtf", help="Enable multi-timeframe HTF feature injection"),
    copy_ticks: bool = typer.Option(False, "--copy-ticks/--no-copy-ticks", help="Copy full tick file into processed dataset"),
) -> None:
    """Align bars and ticks into a canonical dataset with a manifest."""
    from slytrade.backtest.reporting import load_bars_file, load_ticks_file
    from slytrade.data.alignment import align_market_data, render_manifest, save_aligned_dataset

    bars = load_bars_file(Path(bars_file))
    ticks = load_ticks_file(Path(ticks_file))
    dataset = align_market_data(
        bars,
        ticks,
        timeframe=timeframe,
        canonical_symbol=canonical_symbol,
        bar_source=bar_source,
        tick_source=tick_source,
        max_quote_age_seconds=max_quote_age_seconds,
        min_fresh_coverage=min_fresh_coverage,
        include_ict_features=include_ict_features,
        include_tick_features=include_tick_features,
        require_fresh_quotes=require_fresh_quotes,
    )
    manifest = save_aligned_dataset(
        dataset,
        output_dir,
        source_bars_file=bars_file,
        source_ticks_file=ticks_file,
        copy_ticks=copy_ticks,
    )
    render_manifest(manifest, console=console)


@app.command()
def run_aligned_backtest(
    bars_file: str = typer.Option(..., help="Aligned bars file with decision quote columns (.csv or .parquet)"),
    strategy: str = typer.Option("no-trade", help="no-trade, buy-and-hold, ma-cross, or ict-bias"),
    symbol: str | None = typer.Option(None, help="Symbol override if needed"),
    volume: float = typer.Option(0.1, help="Order volume used by baseline strategies"),
    initial_balance: float = typer.Option(100_000.0, help="Initial account balance"),
    point_size: float = typer.Option(0.01, help="Instrument point size"),
    point_value: float = typer.Option(1.0, help="PnL value per price unit and volume"),
    symbol_spec_file: str | None = typer.Option(None, help="Optional symbol spec JSON to set point size/value"),
    slippage_points: float = typer.Option(0.0, help="Adverse slippage in points"),
    commission_per_volume: float = typer.Option(0.0, help="Commission per traded volume unit"),
    fast_window: int = typer.Option(5, help="Fast MA window for ma-cross"),
    slow_window: int = typer.Option(20, help="Slow MA window for ma-cross"),
) -> None:
    """Run a fast backtest from an aligned bars file with precomputed quotes."""
    from slytrade.backtest.reporting import (
        VALID_STRATEGIES,
        load_bars_file,
        render_backtest_report,
        run_aligned_backtest_from_bars,
    )

    if strategy not in VALID_STRATEGIES:
        raise typer.BadParameter(f"strategy must be one of: {', '.join(VALID_STRATEGIES)}")
    bars = load_bars_file(Path(bars_file))
    result = run_aligned_backtest_from_bars(
        bars,
        strategy_name=strategy,
        symbol=symbol,
        volume=volume,
        fast_window=fast_window,
        slow_window=slow_window,
        config=build_backtest_config_from_cli(
            initial_balance=initial_balance,
            point_size=point_size,
            point_value=point_value,
            slippage_points=slippage_points,
            commission_per_volume=commission_per_volume,
            symbol_spec_file=symbol_spec_file,
        ),
    )
    render_backtest_report(result, strategy_name=f"aligned-{strategy}", console=console)


@app.command()
def run_managed_backtest(
    bars_file: str = typer.Option(..., help="Aligned bars file with quote/tick/ICT columns"),
    strategy: str = typer.Option("ict-bias", help="Entry strategy (validated at runtime)"),
    min_mtf_score: int = typer.Option(2, help="Minimum MTF confluence score required"),
    require_mtf_bias_alignment: bool = typer.Option(True, "--require-mtf-bias/--no-require-mtf-bias", help="Require HTF bias alignment"),
    symbol: str | None = typer.Option(None, help="Symbol override if needed"),
    volume: float = typer.Option(0.1, help="Order volume"),
    initial_balance: float = typer.Option(100_000.0, help="Initial account balance"),
    point_size: float = typer.Option(0.01, help="Instrument point size"),
    point_value: float = typer.Option(1.0, help="PnL value per price unit and volume"),
    symbol_spec_file: str | None = typer.Option(None, help="Optional symbol spec JSON to set point size/value"),
    slippage_points: float = typer.Option(0.0, help="Adverse slippage in points"),
    commission_per_volume: float = typer.Option(0.0, help="Commission per traded volume unit"),
    stop_loss_atr: float = typer.Option(1.0, help="Stop-loss distance in ATR multiples"),
    take_profit_atr: float = typer.Option(2.0, help="Final take-profit distance in ATR multiples"),
    min_stop_distance: float = typer.Option(0.10, help="Minimum stop/target distance in price units"),
    max_bars_in_trade: int | None = typer.Option(None, help="Optional time exit in bars"),
    partial_take_profit: bool = typer.Option(False, "--partial-tp/--no-partial-tp", help="Enable partial take-profit"),
    partial_take_profit_atr: float = typer.Option(1.0, help="Partial take-profit distance in ATR multiples"),
    partial_close_fraction: float = typer.Option(0.5, help="Fraction of initial volume to close at partial TP"),
    breakeven_after_partial: bool = typer.Option(True, "--breakeven/--no-breakeven", help="Move SL to breakeven after partial TP"),
    trailing_stop_atr: float | None = typer.Option(None, help="Optional trailing stop distance in ATR multiples"),
    fast_window: int = typer.Option(5, help="Fast MA window for ma-cross"),
    slow_window: int = typer.Option(20, help="Slow MA window for ma-cross"),
) -> None:
    """Run an aligned backtest with basic stop-loss/take-profit management."""
    from slytrade.backtest.reporting import (
        VALID_STRATEGIES,
        load_bars_file,
        render_backtest_report,
        run_managed_aligned_backtest_from_bars,
    )
    from slytrade.backtest.trade_management import TradeManagementConfig

    if strategy not in VALID_STRATEGIES or strategy == "no-trade":
        raise typer.BadParameter("managed backtest strategy must be one of: buy-and-hold, ma-cross, ict-bias, ict-confluence, mtf-ict-confluence")
    bars = load_bars_file(Path(bars_file))
    result = run_managed_aligned_backtest_from_bars(
        bars,
        strategy_name=strategy,
        symbol=symbol,
        volume=volume,
        fast_window=fast_window,
        slow_window=slow_window,
        config=build_backtest_config_from_cli(
            initial_balance=initial_balance,
            point_size=point_size,
            point_value=point_value,
            slippage_points=slippage_points,
            commission_per_volume=commission_per_volume,
            symbol_spec_file=symbol_spec_file,
        ),
        trade_config=TradeManagementConfig(
            stop_loss_atr=stop_loss_atr,
            take_profit_atr=take_profit_atr,
            min_stop_distance=min_stop_distance,
            max_bars_in_trade=max_bars_in_trade,
            partial_take_profit_enabled=partial_take_profit,
            partial_take_profit_atr=partial_take_profit_atr,
            partial_close_fraction=partial_close_fraction,
            move_to_breakeven_after_partial=breakeven_after_partial,
            trailing_stop_atr=trailing_stop_atr,
        ),
    )
    render_backtest_report(result, strategy_name=f"managed-{strategy}", console=console)


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
    symbol_spec_file: str | None = typer.Option(None, help="Optional symbol spec JSON to set point size/value"),
    default_spread_points: float = typer.Option(20.0, help="Fallback spread in points when no tick quote exists"),
    slippage_points: float = typer.Option(0.0, help="Adverse slippage in points"),
    commission_per_volume: float = typer.Option(0.0, help="Commission per traded volume unit"),
    max_quote_age_seconds: float = typer.Option(5.0, help="Maximum fresh quote age at each bar decision time"),
    allow_bar_quote_fallback: bool = typer.Option(
        False,
        "--allow-bar-quote-fallback/--no-bar-quote-fallback",
        help="Allow synthetic bar-close quote when no fresh tick is available",
    ),
    fast_window: int = typer.Option(5, help="Fast MA window for ma-cross"),
    slow_window: int = typer.Option(20, help="Slow MA window for ma-cross"),
) -> None:
    """Run a baseline backtest where bar signals execute on tick bid/ask quotes."""
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
        config=build_backtest_config_from_cli(
            initial_balance=initial_balance,
            default_spread_points=default_spread_points,
            point_size=point_size,
            point_value=point_value,
            slippage_points=slippage_points,
            commission_per_volume=commission_per_volume,
            max_quote_age_seconds=max_quote_age_seconds,
            allow_bar_quote_fallback=allow_bar_quote_fallback,
            symbol_spec_file=symbol_spec_file,
        ),
    )
    render_backtest_report(result, strategy_name=f"tick-{strategy}", console=console)


@app.command()
def mt5_info() -> None:
    """Check MT5 / mt5linux bridge connectivity."""
    mt5 = load_mt5()
    initialize_mt5(mt5)
    try:
        terminal_info = mt5.terminal_info() if hasattr(mt5, "terminal_info") else None
        account_info = mt5.account_info() if hasattr(mt5, "account_info") else None
        symbols = mt5.symbols_get() if hasattr(mt5, "symbols_get") else []
        console.print("[bold green]MT5 bridge initialized[/bold green]")
        console.print(f"Terminal: {terminal_info}")
        console.print(f"Account : {account_info}")
        console.print(f"Symbols : {len(symbols) if symbols is not None else 0}")
    finally:
        shutdown_mt5(mt5)


@app.command()
def resolve_symbols(contains: str = typer.Option("XAU", help="Base symbol or case-insensitive symbol name filter")) -> None:
    """Resolve a base symbol and list matching MT5 symbols."""
    from slytrade.brokers.symbols import list_matching_symbols, resolve_symbol

    mt5 = load_mt5()
    initialize_mt5(mt5)
    try:
        table = Table(title=f"MT5 Symbol resolution for '{contains}'")
        table.add_column("Resolved")
        table.add_column("Exact")
        table.add_column("Description")
        try:
            resolved = resolve_symbol(mt5, contains, select=False)
            table.add_row(f"[bold green]{resolved.resolved}[/bold green]", str(resolved.exact), resolved.description)
        except Exception as exc:
            table.add_row(f"[red]not resolved: {exc}[/red]", "False", "")
        for match in list_matching_symbols(mt5, contains):
            if match.resolved != contains:
                table.add_row(match.resolved, str(match.exact), match.description)
        console.print(table)
    finally:
        shutdown_mt5(mt5)


@app.command()
def inspect_data(
    bars_file: str | None = typer.Option(None, help="Canonical bars file (.csv or .parquet)"),
    ticks_file: str | None = typer.Option(None, help="Canonical ticks file (.csv or .parquet)"),
    timeframe: str | None = typer.Option(None, help="Timeframe override for decision-time alignment, e.g. M1"),
    max_quote_age_seconds: float = typer.Option(5.0, help="Maximum fresh quote age at each bar decision time"),
) -> None:
    """Inspect bars/ticks and print data-quality diagnostics."""
    from slytrade.backtest.reporting import load_bars_file, load_ticks_file
    from slytrade.data.diagnostics import (
        inspect_bars,
        inspect_tick_bar_coverage,
        inspect_ticks,
        render_data_diagnostics,
    )

    bars = load_bars_file(Path(bars_file)) if bars_file else None
    ticks = load_ticks_file(Path(ticks_file)) if ticks_file else None
    bars_diag = inspect_bars(bars, timeframe=timeframe) if bars is not None else None
    ticks_diag = inspect_ticks(ticks) if ticks is not None else None
    coverage = (
        inspect_tick_bar_coverage(bars, ticks, timeframe=timeframe, max_quote_age_seconds=max_quote_age_seconds)
        if bars is not None and ticks is not None
        else None
    )
    render_data_diagnostics(bars=bars_diag, ticks=ticks_diag, coverage=coverage, console=console)


@app.command()
def live() -> None:
    """Live trading placeholder."""
    console.print("[bold red]Live trading is disabled at bootstrap stage.[/bold red]")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
