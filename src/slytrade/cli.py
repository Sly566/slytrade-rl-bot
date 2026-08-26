"""SlyTrade v0.8.1 CLI — ICT/SMC scalping bot (Layers 0-5)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="SlyTrade ICT/SMC scalper v0.8.1")
console = Console()


def _hint_bridge():
    console.print("[yellow]Hint:[/yellow] start MT5 bridge with: bash start_mt5_bridge.sh")


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
@app.command()
def doctor():
    """Check dependencies and data dirs."""
    table = Table(title="SlyTrade Doctor v0.8.1"); table.add_column("Check"); table.add_column("Status")
    for mod in ["numpy", "pandas", "pyarrow", "pydantic", "typer", "rich"]:
        table.add_row(f"required:{mod}", "OK" if importlib.util.find_spec(mod) else "MISSING")
    for opt in ["mt5linux"]:
        table.add_row(f"optional:{opt}", "OK" if importlib.util.find_spec(opt) else "not installed")
    for p in ["data/raw", "data/processed"]:
        try: Path(p).mkdir(parents=True, exist_ok=True); probe=Path(p)/".w"; probe.write_text("ok"); probe.unlink(); s="[green]OK (writable)[/green]"
        except OSError as e: s=f"[red]{e}[/red]"
        table.add_row(f"dir:{p}", s)
    console.print(table)


# --------------------------------------------------------------------------- #
# mt5-info
# --------------------------------------------------------------------------- #
@app.command("mt5-info")
def mt5_info():
    """Check MT5 bridge connectivity."""
    try:
        from mt5linux import MetaTrader5  # type: ignore
        mt5 = MetaTrader5(timeout=int(os.getenv("SLYTRADE_MT5_TIMEOUT", "30")))
        if not mt5.initialize():
            console.print("[red]MT5 initialize() failed[/red]"); _hint_bridge(); raise typer.Exit(1)
        console.print("[green]MT5 connected[/green]")
        console.print(f"Terminal: {mt5.terminal_info()}")
        console.print(f"Account:  {mt5.account_info()}")
        mt5.shutdown()
    except ImportError:
        console.print("[red]mt5linux not installed[/red]"); raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]MT5 error: {e}[/red]"); _hint_bridge(); raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
@app.command("inspect")
def inspect_cmd(raw_symbol: str = typer.Option("XAUUSDm", "--symbol"),
                processed_root: str = typer.Option("data/processed", "--processed-root"),
                raw_root: str = typer.Option("data/raw", "--raw-root")):
    """Inspect processed / aligned / signals / backtest data partitions."""
    from slytrade.config import DataConfig
    from slytrade.data.storage import discover_partitions
    cfg = DataConfig.from_paths(Path(raw_root), Path(processed_root))
    # Processed bars
    console.print("[bold]Processed bars:[/bold]")
    for tf in ["M1","M5","M15","M30","H1","H4","D1","W1"]:
        base = cfg.processed_bars_path / f"symbol={raw_symbol}" / f"timeframe={tf}"
        files = discover_partitions(base, "**/*.parquet")
        rows = 0
        for fp in files:
            try:
                import pyarrow.parquet as pq
                rows += pq.ParquetFile(str(fp)).metadata.num_rows
            except Exception: pass
        if files:
            console.print(f"  {tf:4s}: {len(files):3d} partitions, {rows:>10,} rows")
    # Aligned
    base = cfg.aligned_path / f"symbol={raw_symbol}" / "timeframe=M1"
    afiles = discover_partitions(base, "**/*.parquet")
    arows = sum((lambda fp: (lambda pf: pf.metadata.num_rows if False else __import__("pyarrow.parquet").ParquetFile(str(fp)).metadata.num_rows)(fp))(fp) for fp in afiles)
    console.print(f"[bold]Aligned M1:[/bold] files={len(afiles)} rows≈{arows:,}")
    # Signals
    sp = cfg.signals_path / f"symbol={raw_symbol}" / "signals.parquet"
    console.print("[bold]Signals:[/bold]")
    if sp.exists():
        sdf = pd.read_parquet(sp)
        console.print(f"  {sp}: {len(sdf):,} signals")
        if "grade" in sdf.columns:
            console.print(f"  grades: {sdf['grade'].value_counts().to_dict()}")
    else:
        console.print(f"  not found ({sp})")


# --------------------------------------------------------------------------- #
# process (per-TF feature engineering — Layer 2)
# --------------------------------------------------------------------------- #
@app.command("process")
def process_cmd(raw_symbol: str = typer.Option("XAUUSDm", "--symbol"),
                timeframes: str = typer.Option("M1,M5,M15,M30,H1,H4,D1,W1", "--timeframes"),
                raw_root: str = typer.Option("data/raw", "--raw-root"),
                processed_root: str = typer.Option("data/processed", "--processed-root"),
                clean: bool = typer.Option(False, "--clean")):
    """Run per-TF feature engineering (Layer 2)."""
    from slytrade.config import DataConfig
    from slytrade.data.per_tf import process_all
    cfg = DataConfig.from_paths(Path(raw_root), Path(processed_root))
    tfs = [t.strip() for t in timeframes.split(",") if t.strip()]
    r = process_all(cfg, raw_symbol[:6], raw_symbol, tfs, clean=clean,
                    progress=lambda m: console.print(f"  {m}"))
    for tf, n in r.per_tf_rows.items():
        console.print(f"  {tf}: {n:,} rows / {r.per_tf_files[tf]} files")


# --------------------------------------------------------------------------- #
# align (MTF causal alignment — Layer 3)
# --------------------------------------------------------------------------- #
@app.command("align")
def align_cmd(raw_symbol: str = typer.Option("XAUUSDm", "--symbol"),
              raw_root: str = typer.Option("data/raw", "--raw-root"),
              processed_root: str = typer.Option("data/processed", "--processed-root"),
              clean: bool = typer.Option(False, "--clean")):
    """Causally align HTF features onto M1 (Layer 3)."""
    from slytrade.config import DataConfig
    from slytrade.data.mtf_align import align_all
    cfg = DataConfig.from_paths(Path(raw_root), Path(processed_root))
    r = align_all(cfg, raw_symbol.split("m")[0], raw_symbol, clean=clean,
                  progress=lambda m: console.print(f"  {m}"))
    console.print(f"Aligned: {r.rows:,} M1 rows × {r.columns} cols across {r.files} files")


# --------------------------------------------------------------------------- #
# scan (signal generation — Layer 4)
# --------------------------------------------------------------------------- #
@app.command("scan")
def scan_cmd(raw_symbol: str = typer.Option("XAUUSDm", "--symbol"),
             raw_root: str = typer.Option("data/raw", "--raw-root"),
             processed_root: str = typer.Option("data/processed", "--processed-root"),
             output: Optional[str] = typer.Option(None, "--output")):
    """Scan aligned M1 bars for ICT/SMC signals (Layer 4)."""
    from slytrade.config import DataConfig
    from slytrade.strategy.config import StrategyConfig
    from slytrade.strategy.scanner import scan_aligned, write_signals
    cfg = DataConfig.from_paths(Path(raw_root), Path(processed_root))
    scfg = StrategyConfig()
    r = scan_aligned(cfg, raw_symbol, cfg=scfg, progress=lambda m: console.print(f"  {m}"))
    if output:
        out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
        from slytrade.strategy.signals import signals_to_frame
        sdf = signals_to_frame(r.signals); sdf.to_parquet(out, index=False)
        console.print(f"Wrote {len(sdf):,} signals to {out}")
    else:
        out = write_signals(cfg, raw_symbol, r.signals)
        console.print(f"Wrote {len(r.signals):,} signals to {out}")


# --------------------------------------------------------------------------- #
# backtest (Layer 5)
# --------------------------------------------------------------------------- #
@app.command("backtest")
def backtest_cmd(raw_symbol: str = typer.Option("XAUUSDm", "--symbol"),
                 equity: float = typer.Option(20000.0, "--equity"),
                 signals_path: Optional[str] = typer.Option(None, "--signals"),
                 raw_root: str = typer.Option("data/raw", "--raw-root"),
                 processed_root: str = typer.Option("data/processed", "--processed-root"),
                 output_dir: Optional[str] = typer.Option(None, "--output"),
                 usd_zar: float = typer.Option(18.5, "--usd-zar"),
                 slip_pts: int = typer.Option(5, "--slip-pts"),
                 max_risk: float = typer.Option(0.02, "--max-risk"),
                 commission: float = typer.Option(0.0, "--commission-per-lot")):
    """Run the hedging backtest engine (Layer 5)."""
    import json
    from slytrade.backtest import run_backtest, BacktestConfig, AccountSpec
    from slytrade.config import DataConfig
    from slytrade.strategy.config import StrategyConfig
    cfg = DataConfig.from_paths(Path(raw_root), Path(processed_root))
    sp = Path(signals_path) if signals_path else cfg.signals_path / f"symbol={raw_symbol}" / "signals.parquet"
    if not sp.exists():
        console.print(f"[red]Signals not found at {sp}. Run `slytrade scan` first.[/red]"); raise typer.Exit(1)
    sdf = pd.read_parquet(sp); sdf["time"] = pd.to_datetime(sdf["time"], utc=True)
    console.print(f"Loaded {len(sdf):,} signals from {sp}")
    acct = AccountSpec(starting_equity=equity, currency="ZAR", leverage=2000,
                       fx_to_account={"USD": usd_zar}, commission_per_lot_rt=commission)
    bt_cfg = BacktestConfig(starting_equity=equity, account_ccy="ZAR", leverage=2000,
                            usd_zar=usd_zar, slippage_points_long=slip_pts, slippage_points_short=slip_pts,
                            commission_per_lot_rt=commission, pay_entry_spread=True,
                            max_open_positions=10, min_equity_fraction=0.30, max_risk_per_trade=max_risk)
    result = run_backtest(cfg, raw_symbol, sdf, account=acct, bt_cfg=bt_cfg,
                          strat_cfg=StrategyConfig(), progress=lambda m: console.print(f"  {m}"))
    m = result.metrics
    console.print("\n[bold green]=== BACKTEST RESULTS ===[/bold green]")
    if "error" in m:
        console.print(f"[red]{m['error']}[/red]"); return
    console.print(f"  Bars processed : {result.n_bars:,}")
    console.print(f"  Signals fired  : {result.n_signals:,}")
    console.print(f"  Trades taken   : {m['n_trades']}")
    console.print(f"  Win rate       : {m['win_rate']*100:.1f}% ({m['n_win']}W/{m['n_loss']}L/{m['n_be']}BE)")
    console.print(f"  Profit factor  : [bold]{m['profit_factor']:.2f}[/bold]")
    console.print(f"  Mean R         : {m['mean_R']:+.3f}")
    console.print(f"  Total P&L      : {m['total_pnl']:+,.2f} {bt_cfg.account_ccy}")
    console.print(f"  Final equity   : {m.get('final_equity', equity):,.2f} {bt_cfg.account_ccy}")
    console.print(f"  Return         : {m.get('return_pct',0):+.2f}%")
    console.print(f"  Max drawdown   : {m.get('max_drawdown_acct',0):,.2f} {bt_cfg.account_ccy} ({m.get('max_drawdown_pct',0):.2f}%)")
    # Breakdowns
    for section, label in [("by_grade","By grade"),("by_session","By session"),
                           ("by_zone","By zone kind"),("by_direction","By direction"),
                           ("by_killzone","By killzone"),("by_ob_tf","By OB TF"),("by_exit","By exit reason")]:
        if section in m and m[section]:
            console.print(f"\n  [bold]{label}:[/bold]")
            for k, v in m[section].items():
                pf = v.get("PF", 0); n = v.get("n",0); wr = v.get("win_rate",0)*100 if "win_rate" in v else 0
                pnl = v.get("total_pnl", 0); mr = v.get("mean_R", 0)
                color = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
                console.print(f"    {str(k):14s}: n={n:4d}  wr={wr:5.1f}%  meanR={mr:+.3f}  PF=[{color}]{pf:.2f}[/]  P&L={pnl:+,.0f}")
    # Save outputs
    if output_dir:
        out = Path(output_dir)/f"symbol={raw_symbol}"; out.mkdir(parents=True, exist_ok=True)
        result.trades.to_parquet(out/"trades.parquet", index=False)
        result.tranches.to_parquet(out/"tranches.parquet", index=False)
        result.equity_curve.to_parquet(out/"equity.parquet", index=False)
        (out/"metrics.json").write_text(json.dumps(m, indent=2, default=str))
        console.print(f"\nSaved trades/tranches/equity/metrics to {out}")


if __name__ == "__main__":
    app()
