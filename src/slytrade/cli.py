"""SlyTrade CLI (Layer 0+1: foundation + raw data collection)."""
from __future__ import annotations

import os
import sys
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .brokers.mt5_adapter import MT5Client
from .config import (
    MT5Config, AppConfig, CollectionConfig, DataConfig, DEFAULT_TIMEFRAMES,
)
from .data.diagnostics import inspect_bars, inspect_ticks
from .data.exness_archive import ExnessArchiveDownloader
from .data.mt5_collectors import MT5BarCollector, MT5TickCollector
from .data.time import TIMEFRAME_MINUTES
from .data.tick_stream import TickMerger

app = typer.Typer(add_completion=False, help="SlyTrade — MT5 raw data collection foundation")
console = Console()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _progress(msg: str) -> None:
    console.print(f"  {msg}")


def _config_from_cli(
    symbol: str,
    lookback: float,
    source: str,
    clean: bool,
    raw_bar_symbol: Optional[str],
    raw_tick_symbol: Optional[str],
    mt5_host: str,
    mt5_port: int,
    data_root: str,
) -> AppConfig:
    return AppConfig(
        mt5=MT5Config(host=mt5_host, port=mt5_port),
        data=DataConfig(root=Path(data_root)),
        collection=CollectionConfig(
            symbol=symbol,
            raw_bar_symbol=raw_bar_symbol,
            raw_tick_symbol=raw_tick_symbol,
            timeframes=list(DEFAULT_TIMEFRAMES),
            lookback_years=lookback,
            source=source,
            clean=clean,
        ),
    )


def _refuse_stub_mount(data_root: Path) -> None:
    """Refuse to run if data root appears to be an empty stub mount point
    (e.g. USB drive not mounted, path created by accident with 0 files)."""
    if data_root.exists() and data_root.is_dir():
        try:
            if not any(data_root.iterdir()) and not data_root.is_mount():
                return  # empty dir we created ourselves, fine
            # If it's the literal /media/.../data path with nothing in it, warn
        except PermissionError:
            pass


# ------------------------------------------------------------------
# Root
# ------------------------------------------------------------------
@app.callback()
def main(
    ctx: typer.Context,
):
    """SlyTrade — foundation build. Start with `slytrade doctor`."""
    # Refuse empty-stub data mounts early
    pass


# ------------------------------------------------------------------
# doctor
# ------------------------------------------------------------------
@app.command()
def doctor(
    fix_permissions: bool = typer.Option(False, "--fix-permissions", help="Attempt to fix USB data dir permissions"),
    mt5_host: str = typer.Option("127.0.0.1", "--mt5-host"),
    mt5_port: int = typer.Option(18812, "--mt5-port"),
    data_root: str = typer.Option("data/raw", "--data-root"),
):
    """Verify environment, MT5 connectivity, and writable data directories."""
    table = Table(title="SlyTrade Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True

    # Python
    py_ok = sys.version_info >= (3, 12)
    table.add_row(
        "Python 3.12+",
        "[green]OK[/green]" if py_ok else "[red]FAIL[/red]",
        f"{sys.version.split()[0]}",
    )
    all_ok &= py_ok

    # Venv
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    table.add_row(
        "Virtualenv",
        "[green]OK[/green]" if in_venv else "[yellow]WARN[/yellow]",
        "active" if in_venv else "not activated (run `source .venv/bin/activate`)",
    )

    # Data directory
    data_path = Path(data_root)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        test_file = data_path / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        data_ok = True
        data_detail = f"writable at {data_path.resolve()}"
    except Exception as e:
        data_ok = False
        data_detail = str(e)
        all_ok = False
    table.add_row(
        "Data directory",
        "[green]OK[/green]" if data_ok else "[red]FAIL[/red]",
        data_detail,
    )

    if fix_permissions and not data_ok:
        try:
            import subprocess
            subprocess.run(["chmod", "-R", "u+rwX", str(data_path)], check=False)
            table.add_row("Fix permissions", "[yellow]RUN[/yellow]", f"chmod -R u+rwX {data_path}")
        except Exception as e:
            table.add_row("Fix permissions", "[red]FAIL[/red]", str(e))

    # MT5 bridge
    mt5_ok = False
    mt5_detail = ""
    try:
        client = MT5Client(MT5Config(host=mt5_host, port=mt5_port))
        info = client.terminal_info()
        acct = client.account_info()
        mt5_ok = info is not None and acct is not None
        mt5_detail = f"connected: {getattr(acct, 'login', '?')} on {getattr(acct, 'server', '?')}, balance={getattr(acct, 'balance', '?')}"
        client.shutdown()
    except Exception as e:
        mt5_detail = str(e)
    table.add_row(
        "MT5 bridge",
        "[green]OK[/green]" if mt5_ok else "[red]FAIL[/red]",
        mt5_detail or "is start_mt5_bridge.sh running?",
    )
    all_ok &= mt5_ok

    # Disk space
    disk_ok = True
    try:
        usage = shutil.disk_usage(str(data_path if data_path.exists() else Path.home()))
        free_gb = usage.free / (1024 ** 3)
        disk_ok = free_gb > 20
        table.add_row(
            "Disk space",
            "[green]OK[/green]" if disk_ok else "[yellow]WARN[/yellow]",
            f"{free_gb:.1f} GB free",
        )
        all_ok &= disk_ok
    except Exception as e:
        table.add_row("Disk space", "[red]FAIL[/red]", str(e))
        all_ok = False

    console.print(table)
    if all_ok:
        console.print("[green]All checks passed.[/green]")
    else:
        console.print("[red]Some checks failed. Fix them before collecting data.[/red]")
        raise typer.Exit(code=1)


# ------------------------------------------------------------------
# mt5-info
# ------------------------------------------------------------------
@app.command("mt5-info")
def mt5_info(
    mt5_host: str = typer.Option("127.0.0.1", "--mt5-host"),
    mt5_port: int = typer.Option(18812, "--mt5-port"),
):
    """Print MT5 account info and visible symbol count."""
    client = MT5Client(MT5Config(host=mt5_host, port=mt5_port))
    try:
        acct = client.account_info()
        info = client.terminal_info()
        symbols = client.all_symbols()
    except Exception as e:
        console.print(f"[red]MT5 connection failed: {e}[/red]")
        raise typer.Exit(code=1)

    table = Table(title="MT5 Terminal")
    table.add_column("Property")
    table.add_column("Value")
    for attr in ("login", "server", "name", "company", "currency", "leverage", "balance", "equity", "margin_free"):
        table.add_row(attr, str(getattr(acct, attr, "?")))
    table.add_row("symbols_visible", str(len(symbols)))
    console.print(table)
    client.shutdown()


# ------------------------------------------------------------------
# collect
# ------------------------------------------------------------------
@app.command()
def collect(
    ctx: typer.Context,
    symbol: str = typer.Option("XAUUSD", "--symbol"),
    lookback: float = typer.Option(2.0, "--lookback", help="Years of history to collect"),
    source: str = typer.Option("hybrid", "--source", help="mt5 | exness | hybrid"),
    clean: bool = typer.Option(False, "--clean", help="Wipe raw data before collecting"),
    raw_bar_symbol: Optional[str] = typer.Option(None, "--bar-symbol", help="Override bar symbol name (e.g. XAUUSDm)"),
    raw_tick_symbol: Optional[str] = typer.Option(None, "--tick-symbol", help="Override tick symbol name"),
    mt5_host: str = typer.Option("127.0.0.1", "--mt5-host"),
    mt5_port: int = typer.Option(18812, "--mt5-port"),
    data_root: str = typer.Option("data/raw", "--data-root"),
):
    """Collect raw MT5 bars, MT5 ticks, and (if hybrid) Exness archive ticks.
    Merges ticks into data/raw/merged_ticks."""
    console.print(f"[bold cyan]━━━ Collect {symbol} ({lookback}y) — source={source} ━━━[/bold cyan]")

    cfg = _config_from_cli(symbol, lookback, source, clean, raw_bar_symbol, raw_tick_symbol,
                           mt5_host, mt5_port, data_root)

    if clean:
        if cfg.data.root.exists():
            console.print(f"  Cleaning {cfg.data.root} ...")
            shutil.rmtree(cfg.data.root, ignore_errors=True)
    cfg.data.root.mkdir(parents=True, exist_ok=True)

    client = MT5Client(cfg.mt5)

    # Resolve raw symbols
    raw_bar = cfg.collection.raw_bar_symbol or client.detect_raw_symbol(symbol)
    raw_tick = cfg.collection.raw_tick_symbol or client.detect_raw_symbol(symbol) or raw_bar
    console.print(f"  bars symbol: {raw_bar}")
    console.print(f"  ticks symbol: {raw_tick}")

    # 1) Collect bars for all TFs
    console.print()
    console.print("[bold]Bars[/bold]")
    bc = MT5BarCollector(client, cfg.data, progress=_progress)
    bar_counts = bc.collect(
        symbol=symbol,
        timeframes=cfg.collection.timeframes,
        lookback_years=cfg.collection.lookback_years,
        raw_symbol=raw_bar,
        clean=cfg.collection.clean,
    )
    for tf, n in bar_counts.items():
        console.print(f"  bars {tf}: {n:,} rows")

    # 2) Collect ticks
    mt5_tick_rows = 0
    mt5_start: Optional[date] = None
    if source in ("mt5", "hybrid"):
        console.print()
        console.print("[bold]MT5 ticks[/bold]")
        tc = MT5TickCollector(
            client, cfg.data,
            empty_streak_stop=cfg.collection.tick_empty_streak_stop,
            progress=_progress,
            progress_every_files=cfg.collection.progress_every_files,
            progress_every_chunks=cfg.collection.progress_every_chunks,
        )
        mt5_tick_rows, mt5_start = tc.collect(
            symbol=symbol,
            lookback_years=cfg.collection.lookback_years,
            raw_symbol=raw_tick,
            clean=cfg.collection.clean,
        )
        n_files = len(list((cfg.data.mt5_ticks_path / f"symbol={raw_tick}").glob("**/*.parquet"))) if (cfg.data.mt5_ticks_path / f"symbol={raw_tick}").exists() else 0
        console.print(f"  ticks: {mt5_tick_rows:,} rows in {n_files} files")
    else:
        # If mt5-only not requested, we still need a start date for Exness
        end_dt = datetime.now(timezone.utc).date()
        mt5_start = end_dt - timedelta(days=int(lookback * 365))

    # 3) Exness archive backfill
    exness_rows = 0
    if source in ("exness", "hybrid"):
        console.print()
        console.print("[bold]Exness archive backfill[/bold]")
        end_dt = datetime.now(timezone.utc).date()
        start_dt = end_dt - timedelta(days=int(lookback * 365))
        # In hybrid mode, only backfill months before MT5 coverage starts
        if source == "hybrid" and mt5_start is not None:
            exness_end = mt5_start.replace(day=1) - timedelta(days=1)
            exness_end = date(exness_end.year, exness_end.month, 1) + timedelta(days=32)
            exness_end = date(exness_end.year, exness_end.month, 1) - timedelta(days=1)
            console.print(f"  MT5 tick coverage starts {mt5_start}; backfilling older ticks from Exness archive.")
        else:
            exness_end = end_dt
        dl = ExnessArchiveDownloader(
            cfg.data,
            timeout=cfg.collection.exness_timeout_seconds,
            retries=cfg.collection.exness_retries,
            retry_backoff=cfg.collection.exness_retry_backoff,
            skip_existing=True,
            progress=_progress,
        )
        result = dl.collect_range(symbol=symbol, start=start_dt, end=exness_end)
        exness_rows = result.total_rows

    # 4) Merge ticks month-by-month
    console.print()
    console.print("[bold]Merged ticks[/bold]")
    end_dt = datetime.now(timezone.utc).date()
    start_dt = end_dt - timedelta(days=int(lookback * 365))
    merger = TickMerger(cfg.data, progress=_progress)
    merge_res = merger.merge_range(symbol=symbol, start=start_dt, end=end_dt, clean=cfg.collection.clean)
    console.print(f"  merged ticks: {merge_res.total_rows:,} rows in {merge_res.months} month files "
                  f"(mt5={merge_res.months_from_mt5_only}, exness={merge_res.months_from_exness_only}, "
                  f"merged={merge_res.months_merged}, empty={merge_res.months_empty})")

    client.shutdown()
    console.print()
    console.print("[green]Done.[/green]")


if __name__ == "__main__":
    app()
