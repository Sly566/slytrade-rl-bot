"""SlyTrade CLI — Full ICT/SMC pipeline: collect → process → align → train → backtest → live.

Each command is self-contained and can be run independently.
Full pipeline: slytrade collect && slytrade process && slytrade align && slytrade train && slytrade backtest
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="SlyTrade v1.0 — ICT/SMC scalping pipeline")
console = Console()

VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# collect — MT5 bars (full period) + ticks hybrid
# ---------------------------------------------------------------------------
@app.command()
def collect(
    symbol: str = typer.Option("XAUUSDm", "--symbol", "-s", help="Trading symbol"),
    years: float = typer.Option(5.0, "--years", "-y", help="Years of history to collect"),
    timeframes: str = typer.Option("M1,M5,M15,M30,H1,H4,D1,W1", "--timeframes", "-t"),
    output: str = typer.Option("data/raw", "--output", "-o", help="Output directory"),
    host: str = typer.Option("127.0.0.1", "--host", help="MT5 bridge host"),
    port: int = typer.Option(18812, "--port", help="MT5 bridge port"),
    clean: bool = typer.Option(False, "--clean", help="Remove existing data first"),
):
    """Collect per-TF bars from MT5 (full period) + ticks hybrid (MT5 + Exness).

    Collection strategy:
    - BARS: ALL timeframes from MT5 for the FULL period (5 years)
    - TICKS: MT5 for current year, Exness archive for older periods
    - Already present data is SKIPPED (safe to re-run)

    Example:
        slytrade collect --symbol XAUUSDm --years 5
    """
    from .data.exness_archive import ExnessArchiveDownloader
    from .data.mt5_collectors import MT5BarCollector, MT5TickCollector
    from .data.storage import MarketDataStorage

    console.print(f"[bold]SlyTrade COLLECT v{VERSION}[/bold] symbol={symbol} years={years}")
    tfs = [t.strip() for t in timeframes.split(",") if t.strip()]
    console.print(f"  Timeframes: {tfs}")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        import shutil
        shutil.rmtree(out_dir / symbol, ignore_errors=True)
        console.print("  Cleaned existing data")

    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25))
    console.print(f"  Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

    storage = MarketDataStorage(root=out_dir)
    total_rows = 0

    # Connect to MT5
    console.print(f"\n  [bold]Connecting to MT5...[/bold]")
    try:
        from .live.trader import connect_mt5
        mt5 = connect_mt5(host, port)
        console.print("  MT5 bridge connected")
    except Exception as e:
        console.print(f"  [red]MT5 connection failed: {e}[/red]")
        console.print(f"  [yellow]Start MT5 bridge: bash start_mt5_bridge.sh[/yellow]")
        raise typer.Exit(1)

    # Phase 1: ALL bars from MT5 for FULL period
    console.print(f"\n  [bold]Phase 1: Bars from MT5 (full {years:.0f} years)[/bold]")
    bar_collector = MT5BarCollector(mt5, storage)

    for tf in tfs:
        console.print(f"    Collecting {tf} from MT5 ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})...")
        result = bar_collector.collect(symbol, tf, start, end)
        total_rows += result.rows
        if result.rows > 0:
            console.print(f"      {tf}: {result.rows:,} rows, {result.file_count} files")
        else:
            console.print(f"      {tf}: no new data (already present or empty)")

    # Phase 2: Ticks hybrid — MT5 current year + Exness older
    console.print(f"\n  [bold]Phase 2: Ticks (hybrid MT5 + Exness)[/bold]")
    mt5_tick_start = datetime(end.year, 1, 1, tzinfo=UTC)

    # MT5 ticks for current year
    console.print(f"    MT5 ticks ({mt5_tick_start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})...")
    tick_collector = MT5TickCollector(mt5, storage)
    tick_result = tick_collector.collect(symbol, mt5_tick_start, end)
    total_rows += tick_result.rows
    if tick_result.rows > 0:
        console.print(f"      Ticks: {tick_result.rows:,} rows, {tick_result.file_count} files")
    else:
        console.print(f"      Ticks: no new data")

    # Exness ticks for older period
    if start < mt5_tick_start:
        console.print(f"    Exness ticks ({start.strftime('%Y-%m-%d')} to {mt5_tick_start.strftime('%Y-%m-%d')})...")
        dl = ExnessArchiveDownloader(output_dir=str(out_dir / symbol))
        exness_result = dl.collect(symbol, start, mt5_tick_start)
        total_rows += exness_result.rows
        console.print(f"      Exness: {exness_result.rows:,} rows, {len(exness_result.files)} files")
        if hasattr(exness_result, 'errors') and exness_result.errors:
            for err in exness_result.errors[:5]:
                console.print(f"        [yellow]Warning: {err}[/yellow]")

    mt5.shutdown()

    # Phase 3: News calendar
    console.print(f"\n  [bold]Phase 3: News Calendar[/bold]")
    news_collected = False

    # Method 1: faireconomy.media (free JSON, mirrors ForexFactory)
    try:
        from .data.news import collect_news_from_faireconomy
        console.print(f"    faireconomy.media ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})...")
        fe_events = collect_news_from_faireconomy(start, end, currencies=["USD", "EUR", "GBP", "XAU"])
        if fe_events:
            import json as _json
            news_dir = Path(output) / "news"
            news_dir.mkdir(parents=True, exist_ok=True)
            cache_file = news_dir / f"news_{start.strftime('%Y%m')}_{end.strftime('%Y%m')}.json"
            with open(cache_file, "w") as f:
                _json.dump(fe_events, f, indent=2, default=str)
            high_impact = [e for e in fe_events if e.get("impact", "").lower() in ("high", "red")]
            console.print(f"    News: {len(fe_events):,} events, {len(high_impact):,} high-impact")
            if high_impact:
                for evt in high_impact[:5]:
                    console.print(f"      {evt.get('time', '')} {evt.get('currency', '')} {evt.get('event', '')}")
                if len(high_impact) > 5:
                    console.print(f"      ... and {len(high_impact) - 5} more")
            news_collected = True
        else:
            console.print(f"    faireconomy.media: 0 events for this period")
    except Exception as e:
        console.print(f"    [yellow]faireconomy.media: {e}[/yellow]")

    # Method 2: MT5 calendar fallback
    if not news_collected:
        try:
            from .data.news import collect_news_from_mt5
            console.print(f"    MT5 economic calendar...")
            mt5_events = collect_news_from_mt5(mt5, start, end)
            if mt5_events:
                import json as _json
                news_dir = Path(output) / "news"
                news_dir.mkdir(parents=True, exist_ok=True)
                cache_file = news_dir / f"mt5_calendar_{start.strftime('%Y%m')}_{end.strftime('%Y%m')}.json"
                with open(cache_file, "w") as f:
                    _json.dump(mt5_events, f, indent=2, default=str)
                high_impact = [e for e in mt5_events if e.get("impact", "").lower() in ("high", "red")]
                console.print(f"    MT5 News: {len(mt5_events):,} events, {len(high_impact):,} high-impact")
                news_collected = True
            else:
                console.print(f"    MT5 calendar: no events")
        except Exception as e:
            console.print(f"    [yellow]MT5 calendar: {e}[/yellow]")

    if not news_collected:
        console.print(f"    [yellow]No news data. News features will be zeros.[/yellow]")

    # Summary
    console.print(f"\n[green]Collection complete: {total_rows:,} total rows[/green]")
    console.print(f"Next: [bold]slytrade process --symbol {symbol}[/bold]")


# ---------------------------------------------------------------------------
# process — Compute per-TF features
# ---------------------------------------------------------------------------
@app.command()
def process(
    symbol: str = typer.Option("XAUUSDm", "--symbol", "-s"),
    timeframes: str = typer.Option("M1,M5,M15,M30,H1,H4,D1,W1", "--timeframes", "-t",
                                    help="Comma-separated timeframes"),
    raw_root: str = typer.Option("data/raw", "--raw-root"),
    output: str = typer.Option("data/processed", "--output", "-o"),
    clean: bool = typer.Option(False, "--clean"),
):
    """Process per-TF features (ATR, structure, OBs, FVGs, sweeps, etc.).

    Loads raw bar files per-TF from MT5 collection, computes all ICT/SMC
    features, and writes processed parquet files. Memory-efficient: loads
    one TF at a time.

    Example:
        slytrade process --symbol XAUUSDm --timeframes M1,M5,M15,M30,H1
    """
    from .data.features import DEFAULT_CONFIG, process_bars

    console.print(f"[bold]SlyTrade PROCESS v{VERSION}[/bold] symbol={symbol}")
    tfs = [t.strip() for t in timeframes.split(",") if t.strip()]
    console.print(f"  Timeframes: {tfs}")

    raw_dir = Path(raw_root)
    out_dir = Path(output) / "bars" / f"symbol={symbol}"

    if clean:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)

    for tf in tfs:
        console.print(f"\n  Processing {tf}...")
        t0 = time.time()

        # Load per-TF bar files from mt5_bars/symbol=X/timeframe=Y/
        tf_bar_dir = raw_dir / "mt5_bars" / f"symbol={symbol}" / f"timeframe={tf}"
        tf_files = sorted(tf_bar_dir.rglob("*.parquet"))

        if not tf_files:
            console.print(f"    [yellow]No raw {tf} bars found in {tf_bar_dir}[/yellow]")
            continue

        # Load in chunks to avoid OOM — concat with explicit dtypes
        frames = []
        for f in tf_files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as e:
                console.print(f"    [yellow]Skipping {f.name}: {e}[/yellow]")
        if not frames:
            console.print(f"    [yellow]No valid {tf} data[/yellow]")
            continue

        df = pd.concat(frames, ignore_index=True)
        del frames  # free memory

        df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

        # Ensure required columns
        for col in ["tick_volume", "real_volume"]:
            if col not in df.columns:
                df[col] = df.get("volume", 0.0)

        console.print(f"    {len(df):,} raw {tf} bars ({df['time'].min()} to {df['time'].max()})")

        # Compute features (with tick data for M1)
        console.print(f"    Computing features...")
        tick_dir = Path(raw_root) / "mt5_ticks" / f"symbol={symbol}" if tf == "M1" else None
        processed = process_bars(df, tf, DEFAULT_CONFIG, tick_dir=tick_dir)
        del df  # free memory

        elapsed = time.time() - t0

        # Wire news features for M1 (Gap 5)
        if tf == "M1":
            news_dir = Path(output).parent / "news"
            alt_news_dir = Path("data/news")
            for nd in [news_dir, alt_news_dir]:
                if nd.exists():
                    news_cache_files = sorted(nd.glob("*.json"))
                    if news_cache_files:
                        console.print(f"    Merging news features from {nd}...")
                        from .data.news import create_news_features
                        import json as _json
                        all_events = []
                        for nf in news_cache_files:
                            try:
                                with open(nf) as jf:
                                    all_events.extend(_json.load(jf))
                            except Exception:
                                continue
                        if all_events:
                            news_df = pd.DataFrame(all_events)
                            processed = create_news_features(processed, news_df)
                            console.print(f"    News: {len(news_df)} events merged")
                        break

        # Save
        tf_dir = out_dir / f"timeframe={tf}"
        tf_dir.mkdir(parents=True, exist_ok=True)
        out_path = tf_dir / "data.parquet"
        processed.to_parquet(out_path, index=False)
        console.print(f"    {tf}: {len(processed):,} bars, {len(processed.columns)} cols, "
                      f"{elapsed:.1f}s → {out_path}")
        del processed  # free memory

    console.print(f"\n[green]Processing complete![/green]")
    console.print(f"Next: [bold]slytrade align --symbol {symbol}[/bold]")

# ---------------------------------------------------------------------------
# align — Causal MTF alignment onto M1
# ---------------------------------------------------------------------------
@app.command()
def align(
    symbol: str = typer.Option("XAUUSDm", "--symbol", "-s"),
    processed_root: str = typer.Option("data/processed", "--processed-root"),
    output: str = typer.Option("data/processed/aligned", "--output", "-o"),
    clean: bool = typer.Option(False, "--clean"),
):
    """Causally align HTF features onto M1 execution TF.

    M1 bar at time T sees only HTF information from bars that closed BEFORE T.
    This is the same alignment used by the live trader.

    Example:
        slytrade align --symbol XAUUSDm
    """
    from .data.mtf_align import _asof_merge, _prep_htf_frame
    from .data.time import timeframe_timedelta

    console.print(f"[bold]SlyTrade ALIGN v{VERSION}[/bold] symbol={symbol}")

    proc_dir = Path(processed_root) / "bars" / f"symbol={symbol}"
    if not proc_dir.exists():
        console.print(f"[red]No processed data at {proc_dir}. Run 'slytrade process' first.[/red]")
        raise typer.Exit(1)

    # Load M1
    m1_path = proc_dir / "timeframe=M1" / "data.parquet"
    if not m1_path.exists():
        console.print(f"[red]M1 data not found at {m1_path}[/red]")
        raise typer.Exit(1)

    m1 = pd.read_parquet(m1_path)
    console.print(f"  M1: {len(m1):,} bars, {len(m1.columns)} columns")

    # Load and align HTFs
    htf_tfs = ["M5", "M15", "M30", "H1", "H4", "D1", "W1"]
    df = m1.copy().sort_values("time").reset_index(drop=True)

    for tf in htf_tfs:
        htf_path = proc_dir / f"timeframe={tf}" / "data.parquet"
        if not htf_path.exists():
            console.print(f"  [yellow]{tf}: not found, skipping[/yellow]")
            continue

        htf = pd.read_parquet(htf_path)
        console.print(f"  {tf}: {len(htf):,} bars → aligning...")

        dur = timeframe_timedelta(tf)
        htf = htf.copy()
        htf["decision_time"] = htf["time"] + dur
        prepped = _prep_htf_frame(htf, tf)
        df = _asof_merge(df, prepped, tf)
        console.print(f"    → {len(df):,} rows, {len(df.columns)} columns")

    # Save
    out_dir = Path(output) / f"symbol={symbol}"
    if clean:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aligned.parquet"
    df.to_parquet(out_path, index=False)

    # Summary
    structure_cols = [c for c in df.columns if any(x in c for x in ['disp', 'bos', 'choch', 'sweep', 'ob_', 'fvg_'])]
    console.print(f"\n[green]Aligned: {len(df):,} M1 bars × {len(df.columns)} columns[/green]")
    console.print(f"  Structure features: {len(structure_cols)}")
    for prefix in ["M1_", "M5_", "M15_", "M30_", "H1_", "H4_", "D1_", "W1_"]:
        n = len([c for c in structure_cols if c.startswith(prefix)])
        if n > 0:
            console.print(f"    {prefix.rstrip('_')}: {n}")
    console.print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    console.print(f"\nNext: [bold]slytrade train --symbol {symbol}[/bold] or [bold]slytrade backtest --symbol {symbol}[/bold]")


# ---------------------------------------------------------------------------
# train — Train RL agent with logging + TensorBoard
# ---------------------------------------------------------------------------
@app.command()
def train(
    symbol: str = typer.Option("XAUUSDm", "--symbol", "-s"),
    aligned_path: str = typer.Option("", "--data", "-d", help="Path to aligned parquet (auto-detected if empty)"),
    algo: str = typer.Option("ppo", "--algo", "-a", help="Algorithm: ppo, sac, a2c"),
    timesteps: int = typer.Option(500_000, "--timesteps", "-n"),
    output: str = typer.Option("models", "--output", "-o"),
    max_bars: int = typer.Option(5000, "--max-bars", help="Max bars per episode"),
    tune: bool = typer.Option(False, "--tune", help="Run Optuna hyperparameter search first"),
    log_interval: int = typer.Option(10, "--log-interval", help="Log every N rollouts"),
    tb_log: str = typer.Option("logs/tb", "--tb-log", help="TensorBoard log directory"),
):
    """Train RL agent on aligned data. Agent learns entries AND exits.

    Logs training progress to console and TensorBoard. View live:
        tensorboard --logdir logs/tb

    Example:
        slytrade train --symbol XAUUSDm --timesteps 500000 --algo ppo
    """
    try:
        from stable_baselines3 import PPO, SAC, A2C
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        console.print("[red]stable-baselines3 not installed. Run: pip install 'slytrade-rl-bot[rl]'[/red]")
        raise typer.Exit(1)

    from .rl.env import SlyTradeEnv, OBS_DIM

    # Find aligned data
    partition_files = []
    if not aligned_path:
        candidates = [
            Path("data/processed/aligned") / f"symbol={symbol}" / "aligned.parquet",
            Path("data/aligned") / f"{symbol}_aligned.parquet",
            Path("data/aligned") / f"{symbol}_6m_aligned.parquet",
        ]
        for c in candidates:
            if c.exists():
                aligned_path = str(c)
                break

        # Also check for monthly partitions from align_all()
        if not aligned_path:
            part_dir = Path("data/processed/aligned") / f"symbol={symbol}"
            if part_dir.exists():
                partition_files = sorted(part_dir.rglob("part-*.parquet"))

        if not aligned_path and not partition_files:
            console.print(f"[red]No aligned data found. Run 'slytrade align' first.[/red]")
            raise typer.Exit(1)

    console.print(f"[bold]SlyTrade TRAIN v{VERSION}[/bold] symbol={symbol} algo={algo}")

    # Load aligned data — full dataset, only RL-needed columns
    # Load each partition, immediately filter to needed columns, delete full chunk.
    # Peak memory: one full partition (~170MB) + accumulated filtered frames (~250MB total).
    RL_COLS = [
        "close", "atr_14",
        # M1 structure
        "bull_disp", "bear_disp",
        "minor_bos_up", "minor_bos_dn", "minor_choch_up", "minor_choch_dn",
        # M5 structure
        "M5_bull_disp", "M5_bear_disp",
        "M5_minor_bos_up", "M5_minor_bos_dn", "M5_minor_choch_up", "M5_minor_choch_dn",
        # M15 structure
        "M15_bull_disp", "M15_bear_disp",
        "M15_minor_bos_up", "M15_minor_bos_dn", "M15_minor_choch_up", "M15_minor_choch_dn",
        "M15_major_choch_up", "M15_major_choch_dn",
        # HTF structure (Gap 6)
        "H1_minor_bos_up", "H1_minor_bos_dn", "H1_minor_choch_up", "H1_minor_choch_dn",
        "H4_minor_bos_up", "H4_minor_bos_dn", "H4_minor_choch_up", "H4_minor_choch_dn",
        # D1 structure
        "D1_minor_bos_up", "D1_minor_bos_dn", "D1_minor_choch_up", "D1_minor_choch_dn",
        # W1 structure
        "W1_minor_bos_up", "W1_minor_bos_dn", "W1_minor_choch_up", "W1_minor_choch_dn",
        # Zone proximity
        "ob_proximity", "fvg_proximity", "sweep_proximity",
        # S/R zones
        "sr_support_dist", "sr_resistance_dist",
        "sr_support_count", "sr_resistance_count",
        "at_support", "at_resistance",
        # Supply/Demand zones
        "in_demand_zone", "in_supply_zone",
        "demand_zone_strength", "supply_zone_strength",
        "demand_zone_dist", "supply_zone_dist",
        # Premium/Discount
        "in_premium", "in_discount",
        # ATR regime
        "atr_pct_rank", "atr_expanding", "atr_contracting",
        # Volume
        "tick_vol_ratio", "vol_spike",
        # Liquidity sweeps
        "bull_liq_sweep", "bear_liq_sweep",
        # Tick microstructure
        "tick_buy_ratio", "tick_sell_ratio", "tick_spread_mean",
        "tick_spread_max", "tick_price_velocity",
        "tick_volume_imbalance", "tick_absorption", "tick_count",
        "tick_large_trade_ratio",
        # News features
        "minutes_to_next_high", "minutes_since_last_high",
        "in_news_window", "news_impact_score",
    ]

    if partition_files:
        console.print(f"  Data: {len(partition_files)} monthly partitions (full dataset, {len(RL_COLS)} columns)")
        # Discover columns + time column from first partition
        first = pd.read_parquet(partition_files[0])
        time_col = None
        for candidate in ["time", "time_msc", "datetime", "timestamp"]:
            if candidate in first.columns:
                time_col = candidate
                break
        available = [c for c in RL_COLS if c in first.columns]
        keep_cols = ([time_col] if time_col else []) + available
        # Filter first partition immediately
        first = first[keep_cols].copy()
        frames = [first]
        del first
        # Load remaining partitions with column filter
        for i, pf in enumerate(partition_files[1:], 2):
            chunk = pd.read_parquet(pf, columns=keep_cols)
            frames.append(chunk)
            del chunk
            if i % 12 == 0:
                console.print(f"    Loaded {i}/{len(partition_files)} partitions...")
        df = pd.concat(frames, ignore_index=True)
        del frames
        if time_col:
            df = df.sort_values(time_col).reset_index(drop=True)
            if time_col != "time":
                df = df.rename(columns={time_col: "time"})
    else:
        console.print(f"  Data: {aligned_path}")
        df = pd.read_parquet(aligned_path)
        time_col = None
        for candidate in ["time", "time_msc", "datetime", "timestamp"]:
            if candidate in df.columns:
                time_col = candidate
                break
        available = [c for c in RL_COLS if c in df.columns]
        keep_cols = ([time_col] if time_col else []) + available
        df = df[keep_cols].copy()
        if time_col:
            df = df.sort_values(time_col).reset_index(drop=True)
            if time_col != "time":
                df = df.rename(columns={time_col: "time"})

    console.print(f"  Timesteps: {timesteps:,}")
    console.print(f"  Observation space: {OBS_DIM} dimensions (M1+M5+M15 structure)")
    console.print(f"  {len(df):,} bars, {len(df.columns)} columns")

    # Train/test split (80/20, chronological — no lookahead)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    del df  # free memory
    console.print(f"  Train: {len(train_df):,} bars ({train_df['time'].min()} to {train_df['time'].max()})")
    console.print(f"  Test:  {len(test_df):,} bars ({test_df['time'].min()} to {test_df['time'].max()})")

    # Optuna tuning
    best_params = {}
    if tune:
        try:
            import optuna
        except ImportError:
            console.print("[yellow]optuna not installed, skipping tuning[/yellow]")
            tune = False

        if tune:
            console.print("\n  Running Optuna hyperparameter search (50 trials)...")

            def objective(trial):
                lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
                gamma = trial.suggest_float("gamma", 0.95, 0.999)
                ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
                n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
                net_arch = trial.suggest_categorical("net_arch", [
                    [128, 128], [256, 128], [256, 256], [128, 64],
                ])

                def make_env():
                    return Monitor(SlyTradeEnv(train_df, max_bars=max_bars))

                env = DummyVecEnv([make_env])
                policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))
                model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma,
                           ent_coef=ent_coef, n_steps=n_steps, policy_kwargs=policy_kwargs,
                           verbose=0, seed=42)
                model.learn(total_timesteps=50_000)

                eval_env = SlyTradeEnv(test_df, max_bars=len(test_df))
                obs, _ = eval_env.reset()
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, term, trunc, _ = eval_env.step(action)
                    if term or trunc:
                        break
                m = eval_env.get_metrics()
                return m.get("sharpe_ratio", 0) + m.get("win_rate", 0) * 2 - m.get("max_drawdown", 1) * 5

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50, show_progress_bar=True)
            best_params = study.best_params
            console.print(f"  Best params: {json.dumps(best_params, indent=2)}")

    # Create environment
    def make_env():
        return Monitor(SlyTradeEnv(train_df, max_bars=max_bars))

    train_env = DummyVecEnv([make_env])

    # Create model
    # Multi-agent mode
    if algo.lower() == "multi":
        from .rl.multi_agent import MultiAgentEnsemble
        from .rl.multi_train import MultiAgentTrainer

        console.print(f"  Mode: Multi-Agent (9 sub-agents + meta-agent)")
        ensemble = MultiAgentEnsemble()
        console.print(f"  Parameters: {ensemble.total_parameters:,}")

        # TensorBoard log dir for multi-agent
        tb_dir = str(Path(tb_log) / f"multi_{symbol}")
        os.makedirs(tb_dir, exist_ok=True)
        console.print(f"  TensorBoard: {tb_dir}")
        console.print(f"  View live: tensorboard --logdir {tb_log}")

        trainer = MultiAgentTrainer(
            env=SlyTradeEnv(train_df, max_bars=max_bars),
            lr=3e-4, gamma=0.99, n_steps=4096, batch_size=512, n_epochs=10,
            tb_log_dir=tb_dir,
        )

        def progress(msg):
            console.print(msg)

        eval_env = SlyTradeEnv(test_df, max_bars=len(test_df))
        metrics = trainer.train(
            total_timesteps=timesteps,
            eval_env=eval_env,
            eval_freq=100_000,
            output_dir=output,
            symbol=symbol,
            progress_fn=progress,
        )

        console.print(f"\n{'='*60}")
        console.print("[bold green]MULTI-AGENT TRAINING COMPLETE[/bold green]")
        if metrics:
            console.print(f"  Trades:      {metrics.get('n_trades', 0)}")
            console.print(f"  Win Rate:    {metrics.get('win_rate', 0):.1%}")
            console.print(f"  Sharpe:      {metrics.get('sharpe_ratio', 0):.2f}")
            console.print(f"  Max DD:      {metrics.get('max_drawdown', 0):.1%}")
        console.print(f"  Model: {output}/multi_{symbol}_best.pt")
        console.print(f"  TensorBoard: {tb_dir}")

        # Show explainability on a sample observation
        console.print(f"\n[bold]Explainability Demo:[/bold]")
        sample_obs, _ = eval_env.reset()
        sample_tensor = torch.FloatTensor(sample_obs).unsqueeze(0)
        explanation = trainer.ensemble.explain(sample_tensor)
        console.print(f"  {explanation['reasoning']}")

        console.print(f"\nNext: [bold]slytrade backtest --symbol {symbol}[/bold]")
        console.print(f"      [bold]tensorboard --logdir {tb_log}[/bold]")
        return

    algo_cls = {"ppo": PPO, "sac": SAC, "a2c": A2C}[algo.lower()]

    # TensorBoard log dir
    tb_dir = str(Path(tb_log) / f"{algo}_{symbol}")
    os.makedirs(tb_dir, exist_ok=True)

    model_kwargs = {
        "policy": "MlpPolicy", "env": train_env,
        "learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "n_steps": 4096, "batch_size": 512, "n_epochs": 10,
        "policy_kwargs": dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])),
        "verbose": 1, "seed": 42,
        "tensorboard_log": tb_dir,
    }
    if best_params:
        model_kwargs.update(best_params)
        if "net_arch" in best_params:
            model_kwargs["policy_kwargs"] = dict(
                net_arch=dict(pi=best_params["net_arch"], vf=best_params["net_arch"])
            )
            del model_kwargs["net_arch"]

    model = algo_cls(**model_kwargs)
    n_params = sum(p.numel() for p in model.policy.parameters())
    console.print(f"\n  Model: {algo.upper()} ({n_params:,} parameters)")
    console.print(f"  TensorBoard: {tb_dir}")
    console.print(f"  Log interval: every {log_interval} rollouts")
    console.print(f"  Training...\n")

    # Custom callback for rich console logging + eval
    class TrainProgressCallback(BaseCallback):
        def __init__(self, eval_df, eval_freq=100_000, verbose=1):
            super().__init__(verbose)
            self.eval_df = eval_df
            self.eval_freq = eval_freq
            self.best_sharpe = -999
            self.start_time = time.time()
            self.last_eval = 0

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            # Log training metrics from SB3 logger
            if self.num_timesteps - self.last_eval >= self.eval_freq:
                self.last_eval = self.num_timesteps
                self._run_eval()

        def _run_eval(self):
            eval_env = SlyTradeEnv(self.eval_df, max_bars=len(self.eval_df))
            obs, _ = eval_env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = eval_env.step(action)
                if term or trunc:
                    break
            m = eval_env.get_metrics()
            elapsed = time.time() - self.start_time

            # Log to SB3/TensorBoard
            self.logger.record("eval/sharpe", m.get("sharpe_ratio", 0))
            self.logger.record("eval/win_rate", m.get("win_rate", 0))
            self.logger.record("eval/total_pnl", m.get("total_pnl", 0))
            self.logger.record("eval/max_drawdown", m.get("max_drawdown", 0))
            self.logger.record("eval/profit_factor", m.get("profit_factor", 0))
            self.logger.record("eval/n_trades", m.get("n_trades", 0))

            console.print(f"    [{self.num_timesteps:,}/{timesteps:,}] "
                          f"trades={m['n_trades']} wr={m['win_rate']:.0%} "
                          f"pnl={m['total_pnl']:+.0f} sharpe={m['sharpe_ratio']:.2f} "
                          f"dd={m['max_drawdown']:.0%} pf={m['profit_factor']:.2f} "
                          f"({elapsed:.0f}s)")

            if m['sharpe_ratio'] > self.best_sharpe and m['n_trades'] > 10:
                self.best_sharpe = m['sharpe_ratio']
                self.model.save(f"{output}/{algo}_{symbol}_best")
                console.print(f"      [green]New best! Saved.[/green]")

    # Train
    os.makedirs(output, exist_ok=True)
    start_time = time.time()

    callback = TrainProgressCallback(
        eval_df=test_df,
        eval_freq=50_000,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        log_interval=log_interval,
        progress_bar=True,
    )

    # Save final
    model.save(f"{output}/{algo}_{symbol}_final")

    # Final evaluation
    console.print(f"\n{'='*60}")
    console.print("[bold green]FINAL EVALUATION[/bold green]")
    eval_env = SlyTradeEnv(test_df, max_bars=len(test_df))
    obs, _ = eval_env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = eval_env.step(action)
        if term or trunc:
            break
    m = eval_env.get_metrics()
    console.print(f"  Trades:      {m['n_trades']}")
    console.print(f"  Win Rate:    {m['win_rate']:.1%}")
    console.print(f"  Total P&L:   {m['total_pnl']:+.2f}")
    console.print(f"  Sharpe:      {m['sharpe_ratio']:.2f}")
    console.print(f"  Profit Fac:  {m['profit_factor']:.2f}")
    console.print(f"  Max DD:      {m['max_drawdown']:.1%}")
    console.print(f"  Avg Win:     {m['avg_win']:+.2f}")
    console.print(f"  Avg Loss:    {m['avg_loss']:+.2f}")

    with open(f"{output}/metrics.json", "w") as f:
        json.dump(m, f, indent=2, default=str)
    console.print(f"\n  Model: {output}/{algo}_{symbol}_best.zip")
    console.print(f"  Metrics: {output}/metrics.json")
    console.print(f"  TensorBoard: {tb_dir}")
    console.print(f"  Total time: {time.time()-start_time:.0f}s")
    console.print(f"\nNext: [bold]slytrade backtest --symbol {symbol}[/bold] or [bold]slytrade live --symbol {symbol}[/bold]")


# ---------------------------------------------------------------------------
# backtest — Run backtest exactly like live
# ---------------------------------------------------------------------------
@app.command()
def backtest(
    symbol: str = typer.Option("XAUUSDm", "--symbol", "-s"),
    aligned_path: str = typer.Option("", "--data", "-d"),
    equity: float = typer.Option(20000.0, "--equity", "-e"),
    risk_cap: float = typer.Option(0.05, "--risk-cap"),
    working_lot: float = typer.Option(0.04, "--working-lot"),
    max_open: int = typer.Option(3, "--max-open"),
    usd_zar: float = typer.Option(18.5, "--usd-zar"),
    output: str = typer.Option("data/backtest", "--output", "-o"),
    unrestricted: bool = typer.Option(False, "--all", help="All signals (unrestricted persona)"),
    model: str = typer.Option("", "--model", "-m", help="Path to trained RL model (multi-agent .pt or SB3 .zip)"),
    algo: str = typer.Option("multi", "--algo", help="Algorithm: multi, ppo, sac, a2c"),
):
    """Run backtest using the same engine as live trading.

    Uses the same signal pipeline, same exit logic (hybrid ladder + trailing),
    same risk management as the live trader.

    Example:
        slytrade backtest --symbol XAUUSDm --equity 20000 --risk-cap 0.05
    """
    from .backtest.specs import AccountSpec, spec_for_symbol
    from .strategy.config import champion_persona, rl_training_persona
    from .strategy.signals import _evaluate_row

    console.print(f"[bold]SlyTrade BACKTEST v{VERSION}[/bold] symbol={symbol}")

    # Find aligned data
    partition_files = []
    if not aligned_path:
        candidates = [
            Path("data/processed/aligned") / f"symbol={symbol}" / "aligned.parquet",
            Path("data/aligned") / f"{symbol}_aligned.parquet",
            Path("data/aligned") / f"{symbol}_6m_aligned.parquet",
        ]
        for c in candidates:
            if c.exists():
                aligned_path = str(c)
                break

        # Also check for monthly partitions from align_all()
        if not aligned_path:
            part_dir = Path("data/processed/aligned") / f"symbol={symbol}"
            if part_dir.exists():
                partition_files = sorted(part_dir.rglob("part-*.parquet"))

        if not aligned_path and not partition_files:
            console.print(f"[red]No aligned data found. Run 'slytrade align' first.[/red]")
            raise typer.Exit(1)

    if partition_files:
        console.print(f"  Data: {len(partition_files)} monthly partitions")
        frames = [pd.read_parquet(f) for f in partition_files]
        df = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
        del frames
    else:
        console.print(f"  Data: {aligned_path}")
        df = pd.read_parquet(aligned_path)

    console.print(f"  {len(df):,} bars, {len(df.columns)} columns")
    console.print(f"  Date: {df['time'].min()} to {df['time'].max()}")

    # Config (same as live)
    cfg = rl_training_persona() if unrestricted else champion_persona()
    spec = spec_for_symbol(symbol)
    acct = AccountSpec(
        starting_equity=equity, currency="ZAR", leverage=2000,
        fx_to_account={"USD": usd_zar},
    )

    # Load RL model if specified
    rl_filter = None
    if model:
        console.print(f"  Loading RL model: {model}")
        if algo == "multi":
            from .rl.serve import MultiAgentFilter
            rl_filter = MultiAgentFilter(model)
            console.print(f"  Multi-agent model loaded ({rl_filter.ensemble.total_parameters:,} params)")
        else:
            from .rl.serve import RLFilter
            rl_filter = RLFilter(model, algo=algo)
            console.print(f"  SB3 model loaded ({algo.upper()})")

    # Walk bars exactly like live
    console.print(f"\n  Walking {len(df):,} bars...")
    state: dict = {}
    equity_curve = [equity]
    trades: list[dict] = []
    pos_dir = 0
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_lots = 0.0
    pos_bars = 0
    pos_grade = ""
    pos_risk_per_unit = 0.0
    pos_best_price = 0.0
    pos_trail_active = False
    time_stop_bars = 60
    signals_fired: set[str] = set()
    n_signals = 0
    t0 = time.time()

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row.get("close", 0.0))
        atr = float(row.get("atr_14", 0.0)) if pd.notna(row.get("atr_14")) else 0.0

        # Check SL/TP/time-stop on open position
        if pos_dir != 0:
            pos_bars += 1
            r_unit = pos_risk_per_unit if pos_risk_per_unit > 0 else abs(pos_tp - pos_entry)
            cur_r = (price - pos_entry) / r_unit if pos_dir == 1 else (pos_entry - price) / r_unit

            # Track best price for trailing
            if pos_dir == 1:
                pos_best_price = max(pos_best_price, price)
            else:
                pos_best_price = min(pos_best_price, price) if pos_best_price > 0 else price

            # SL hit
            hit_sl = (pos_dir == 1 and price <= pos_sl) or (pos_dir == -1 and price >= pos_sl)
            # TP hit
            hit_tp = (pos_dir == 1 and price >= pos_tp) or (pos_dir == -1 and price <= pos_tp)
            # Time stop
            hit_time = pos_bars >= time_stop_bars

            if hit_sl or hit_tp or hit_time:
                reason = "SL" if hit_sl else ("TP" if hit_tp else "TIME_STOP")
                pnl_pts = (price - pos_entry) * pos_dir
                pnl_quote = pnl_pts * pos_lots * spec.contract_size
                pnl_acct = acct.to_account_ccy(pnl_quote, spec.currency_profit)
                equity += pnl_acct
                equity_curve.append(equity)
                trades.append({
                    "entry": pos_entry, "exit": price, "dir": pos_dir,
                    "lots": pos_lots, "pnl": pnl_acct, "bars": pos_bars,
                    "grade": pos_grade, "reason": reason, "r": cur_r,
                })
                pos_dir = 0
                continue

            # C-grade trailing (same as live)
            if pos_grade == 'C' and pos_trail_active and atr > 0:
                trail_dist = max(0.5 * atr, 0.3 * r_unit)
                if pos_dir == 1:
                    new_trail = pos_best_price - trail_dist
                    if new_trail > pos_sl:
                        pos_sl = new_trail
                else:
                    new_trail = pos_best_price + trail_dist
                    if new_trail < pos_sl:
                        pos_sl = new_trail

            # Activate C-grade trailing at 0.3R
            if pos_grade == 'C' and not pos_trail_active and cur_r >= 0.3:
                pos_trail_active = True
                pos_sl = pos_entry + pos_dir * 0.1 * r_unit

        # Evaluate signal (same as live)
        try:
            sig = _evaluate_row(i, row, cfg, state)
        except Exception:
            sig = None

        if sig is not None:
            n_signals += 1
            side = "LONG" if sig.direction == 1 else "SHORT"
            setup = getattr(sig, 'setup_kind', 'RETEST_OB')
            zone_id = sig.ob_tf or (f"fvg{sig.fvg_top:.0f}" if sig.fvg_top else "-")
            key = f"{sig.time.isoformat()}|{sig.direction}|{sig.grade}|{setup}|{zone_id}"

            if key not in signals_fired:
                # Netting: close opposite
                if pos_dir != 0 and pos_dir == -sig.direction:
                    pnl_pts = (price - pos_entry) * pos_dir
                    pnl_quote = pnl_pts * pos_lots * spec.contract_size
                    pnl_acct = acct.to_account_ccy(pnl_quote, spec.currency_profit)
                    equity += pnl_acct
                    equity_curve.append(equity)
                    trades.append({
                        "entry": pos_entry, "exit": price, "dir": pos_dir,
                        "lots": pos_lots, "pnl": pnl_acct, "bars": pos_bars,
                        "grade": pos_grade, "reason": "NETTING_FLIP", "r": 0,
                    })
                    pos_dir = 0

                # Enter if flat
                if pos_dir == 0:
                    risk_per_unit = abs(price - float(sig.stop))
                    if risk_per_unit > 0:
                        lots = 0.01 if sig.grade == 'C' else working_lot
                        sl = float(sig.stop)
                        # Enforce min SL distance
                        min_dist = max(spec.point * 500, 0.75 * atr) if atr > 0 else spec.point * 500
                        sl_dist = abs(price - sl)
                        if sl_dist < min_dist:
                            sl = price - sig.direction * min_dist
                            risk_per_unit = abs(price - sl)

                        tp = price + sig.direction * cfg.exits.tp1_r * risk_per_unit
                        pos_dir = sig.direction
                        pos_entry = price
                        pos_sl = sl
                        pos_tp = tp
                        pos_lots = lots
                        pos_bars = 0
                        pos_grade = sig.grade
                        pos_risk_per_unit = risk_per_unit
                        pos_best_price = price
                        pos_trail_active = False
                        signals_fired.add(key)

    # Close any remaining position
    if pos_dir != 0:
        price = float(df.iloc[-1].get("close", 0.0))
        pnl_pts = (price - pos_entry) * pos_dir
        pnl_quote = pnl_pts * pos_lots * spec.contract_size
        pnl_acct = acct.to_account_ccy(pnl_quote, spec.currency_profit)
        equity += pnl_acct
        trades.append({
            "entry": pos_entry, "exit": price, "dir": pos_dir,
            "lots": pos_lots, "pnl": pnl_acct, "bars": pos_bars,
            "grade": pos_grade, "reason": "END", "r": 0,
        })

    elapsed = time.time() - t0

    # Results
    console.print(f"\n{'='*60}")
    console.print(f"[bold green]BACKTEST RESULTS[/bold green] ({elapsed:.1f}s)")
    console.print(f"{'='*60}")

    if not trades:
        console.print("[yellow]No trades taken[/yellow]")
        return

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    total_pnl = tdf["pnl"].sum()
    win_rate = len(wins) / len(tdf) if len(tdf) > 0 else 0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    pf = abs(wins["pnl"].sum() / losses["pnl"].sum()) if len(losses) > 0 and losses["pnl"].sum() != 0 else float("inf")

    # Sharpe
    returns = tdf["pnl"].values / equity
    sharpe = np.mean(returns) / max(np.std(returns), 1e-9) * np.sqrt(252 * 24) if len(returns) > 1 else 0

    # Max drawdown
    ec = np.array(equity_curve)
    peak = np.maximum.accumulate(ec)
    dd = (peak - ec) / peak
    max_dd = float(np.max(dd))

    console.print(f"  Signals:     {n_signals}")
    console.print(f"  Trades:      {len(tdf)}")
    console.print(f"  Win Rate:    {win_rate:.1%} ({len(wins)}W/{len(losses)}L)")
    console.print(f"  Avg Win:     {avg_win:+.2f}")
    console.print(f"  Avg Loss:    {avg_loss:+.2f}")
    console.print(f"  Profit Fac:  {pf:.2f}")
    console.print(f"  Sharpe:      {sharpe:.2f}")
    console.print(f"  Total P&L:   {total_pnl:+.2f} ZAR")
    console.print(f"  Return:      {(equity - acct.starting_equity) / acct.starting_equity:.1%}")
    console.print(f"  Max DD:      {max_dd:.1%}")
    console.print(f"  Final Equity:{equity:.2f} ZAR")

    # By grade
    if "grade" in tdf.columns:
        console.print(f"\n  [bold]By Grade:[/bold]")
        for grade, gdf in tdf.groupby("grade"):
            gw = gdf[gdf["pnl"] > 0]
            gl = gdf[gdf["pnl"] <= 0]
            g_wr = len(gw) / len(gdf) if len(gdf) > 0 else 0
            g_pf = abs(gw["pnl"].sum() / gl["pnl"].sum()) if len(gl) > 0 and gl["pnl"].sum() != 0 else float("inf")
            console.print(f"    {grade}: n={len(gdf)} wr={g_wr:.0%} pf={g_pf:.2f} pnl={gdf['pnl'].sum():+.0f}")

    # Save
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tdf.to_parquet(out_dir / "trades.parquet", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "n_trades": len(tdf), "win_rate": win_rate, "profit_factor": pf,
            "sharpe": sharpe, "total_pnl": total_pnl, "max_drawdown": max_dd,
            "avg_win": avg_win, "avg_loss": avg_loss, "final_equity": equity,
        }, f, indent=2)
    console.print(f"\n  Saved: {output}/trades.parquet, {output}/metrics.json")


# ---------------------------------------------------------------------------
# live — Real-time trading
# ---------------------------------------------------------------------------
@app.command()
def live(
    symbol: str = typer.Option("XAUUSDm", "--symbol", "-s"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(18812, "--port"),
    live_mode: bool = typer.Option(False, "--live", help="Send real orders (default: dry-run)"),
    risk_cap: float = typer.Option(0.05, "--risk-cap"),
    working_lot: float = typer.Option(0.04, "--working-lot"),
    max_open: int = typer.Option(3, "--max-open"),
    usd_zar: float = typer.Option(18.5, "--usd-zar"),
    leverage: int = typer.Option(2000, "--leverage"),
    verbose: bool = typer.Option(False, "--verbose"),
    unrestricted: bool = typer.Option(False, "--all"),
    model: str = typer.Option("", "--model", "-m", help="Path to trained RL model"),
    algo: str = typer.Option("multi", "--algo", help="Algorithm: multi, ppo, sac, a2c"),
):
    """Run live trading loop (dry-run or real money).

    Connects to MT5 via RPyC bridge, processes M1 bars in real-time,
    and executes trades using the same signal pipeline as backtest.

    Example:
        slytrade live --symbol XAUUSDm --risk-cap 0.05 --working-lot 0.04 --max-open 3 --all --verbose --live
    """
    from .backtest.specs import AccountSpec
    from .live.trader import LiveTrader, connect_mt5, resolve_symbol_spec
    from .strategy.config import champion_persona, rl_training_persona

    console.print(f"[bold]SlyTrade LIVE v{VERSION}[/bold] symbol={symbol} live={live_mode}")
    mt5 = connect_mt5(host, port)

    def _to_dict(o):
        if o is None: return {}
        if isinstance(o, dict): return o
        try: return o._asdict()
        except Exception: return {k: getattr(o, k) for k in dir(o) if not k.startswith("_")}

    acc = _to_dict(mt5.account_info())
    console.print(f"  login={acc.get('login')} server={acc.get('server')} "
                  f"balance={acc.get('balance')} equity={acc.get('equity')} {acc.get('currency')}")
    resolved, spec, stop_level_pts = resolve_symbol_spec(mt5, symbol, str(acc.get("currency", "ZAR")), usd_zar)
    console.print(f"  symbol={resolved} point={spec.point} digits={spec.digits} "
                  f"contract={spec.contract_size} vol_min={spec.volume_min} stop_level={stop_level_pts}pts")

    acct_spec = AccountSpec(
        starting_equity=float(acc.get("equity", 1000)),
        currency=str(acc.get("currency", "ZAR")),
        leverage=int(acc.get("leverage", leverage)),
        fx_to_account={"USD": usd_zar} if str(acc.get("currency", "ZAR")) != "USD" else {"USD": 1.0},
    )
    cfg = rl_training_persona() if unrestricted else champion_persona()
    trader = LiveTrader(
        mt5=mt5, symbol=resolved, spec=spec, cfg=cfg, acct=acct_spec,
        live=live_mode, risk_cap=risk_cap, working_lot=working_lot,
        max_open=max_open, verbose=verbose, stop_level_pts=stop_level_pts,
    )

    # Load RL model if specified
    if model:
        console.print(f"  Loading RL model: {model}")
        if algo == "multi":
            from .rl.serve import MultiAgentFilter
            trader._rl_filter = MultiAgentFilter(model)
            console.print(f"  Multi-agent RL filter active ({trader._rl_filter.ensemble.total_parameters:,} params)")
        else:
            from .rl.serve import RLFilter
            trader._rl_filter = RLFilter(model, algo=algo)
            console.print(f"  SB3 RL filter active ({algo.upper()})")
    try:
        trader.run()
    finally:
        mt5.shutdown()


# ---------------------------------------------------------------------------
# doctor — Health check
# ---------------------------------------------------------------------------
@app.command()
def news(
    symbol: str = typer.Option("XAUUSDm", "--symbol", "-s"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(18812, "--port"),
    years: float = typer.Option(5.0, "--years", "-y"),
    output: str = typer.Option("data/news", "--output", "-o"),
):
    """Fetch economic calendar news data.

    Tries MT5 broker calendar first (requires MT5 bridge),
    then falls back to Forex Factory scraping.

    Example:
        slytrade news --symbol XAUUSDm --years 5
    """
    import json as _json
    from datetime import UTC, datetime, timedelta

    console.print(f"[bold]SlyTrade NEWS v{VERSION}[/bold]")
    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25))
    console.print(f"  Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Method 1: faireconomy.media
    console.print(f"\n  [bold]News Sources[/bold]")
    try:
        from .data.news import collect_news_from_faireconomy
        console.print(f"  faireconomy.media ({news_start} to {news_end})...")
        events = collect_news_from_faireconomy(start, end, currencies=["USD", "EUR", "GBP", "XAU"])
        if events:
            cache_file = out_dir / f"news_{start.strftime('%Y%m')}_{end.strftime('%Y%m')}.json"
            with open(cache_file, "w") as f:
                _json.dump(events, f, indent=2, default=str)
            high_impact = [e for e in events if e.get("impact", "").lower() in ("high", "red")]
            console.print(f"  MT5: {len(events):,} events, {len(high_impact):,} high-impact")
            console.print(f"  Saved: {cache_file}")
            if high_impact:
                for evt in high_impact[:10]:
                    console.print(f"    {evt.get('time', '')} {evt.get('currency', '')} {evt.get('event', '')} [{evt.get('impact', '')}]")
            return
        else:
            console.print(f"  MT5 calendar: 0 events (may not be supported on this broker)")
    except Exception as e:
        console.print(f"  [yellow]MT5 calendar: {e}[/yellow]")

    # Method 2: Forex Factory
    console.print(f"\n  [bold]Forex Factory (fallback)[/bold]")
    try:
        from .data.news import ForexFactoryCalendar
        ff = ForexFactoryCalendar()
        events = ff.get_events(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), currencies=["USD", "EUR", "GBP"])
        if events:
            console.print(f"  Forex Factory: {len(events)} events")
            return
        else:
            console.print(f"  [yellow]Forex Factory: 0 events (page is JS-rendered)[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]Forex Factory: {e}[/yellow]")

    console.print(f"\n[yellow]No news data available from either source.[/yellow]")
    console.print(f"[yellow]The bot will trade without news awareness.[/yellow]")


@app.command()
def doctor():
    """Check dependencies and environment."""
    table = Table(title=f"SlyTrade Doctor v{VERSION}")
    table.add_column("Check")
    table.add_column("Status")

    for mod in ["numpy", "pandas", "pyarrow", "pydantic", "typer", "rich"]:
        table.add_row(f"required:{mod}", "OK" if importlib.util.find_spec(mod) else "[red]MISSING[/red]")
    for mod in ["mt5linux", "gymnasium", "stable_baselines3", "torch", "optuna"]:
        table.add_row(f"optional:{mod}", "OK" if importlib.util.find_spec(mod) else "not installed")
    for p in ["data/raw", "data/processed", "models"]:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
            probe = Path(p) / ".w"
            probe.write_text("ok")
            probe.unlink()
            table.add_row(f"dir:{p}", "[green]OK[/green]")
        except OSError as e:
            table.add_row(f"dir:{p}", f"[red]{e}[/red]")
    console.print(table)


if __name__ == "__main__":
    app()
