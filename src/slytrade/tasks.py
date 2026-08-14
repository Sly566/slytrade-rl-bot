"""High-level, task-oriented operations for the full trading pipeline.

These functions are the engine behind both the CLI commands and the Rich GUI.
Each task does the whole job (collect, align, backtest, train, walk-forward,
promote, paper, demo) with sensible defaults so a user never has to assemble
flags by hand. They return small result dicts the UI can render.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

console = Console()

# Root directory for synthetic sample data (overridable in tests).
SAMPLE_ROOT = "data/samples"

# Root for bars derived from Exness ticks in the tick-only fallback path.
EXNESS_DERIVED_ROOT = "data/exness_derived"

# ---------------------------------------------------------------------------
# Small result helpers
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    ok: bool
    message: str = ""
    data: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "message": self.message, **(self.data or {})}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def mt5_available() -> bool:
    return _module_available("MetaTrader5") or _module_available("mt5linux")


# ---------------------------------------------------------------------------
# Data location (partitioned storage -> single frames)
# ---------------------------------------------------------------------------


def _symbol_dir(root: str | Path, prefix: str, symbol: str) -> Path | None:
    """Find a partitioned ``symbol=...`` directory for a base symbol.

    MT5 stores data under the *resolved* broker symbol (e.g. ``XAUUSDm``), while
    the CLI and Exness archive use the base symbol (``XAUUSD``). Match the exact
    symbol first, then the shortest suffix variant (``XAUUSDm`` before
    ``XAUUSD247m``), so collection stays autonomous.
    """
    base = Path(root) / prefix
    if not base.exists():
        return None
    exact = base / f"symbol={symbol}"
    if exact.exists():
        return exact
    candidates = sorted(
        (d for d in base.iterdir() if d.is_dir() and d.name.startswith(f"symbol={symbol}")),
        key=lambda d: len(d.name),
    )
    return candidates[0] if candidates else None


def find_collected_bars(symbol: str, timeframe: str, root: str | Path = "data/raw") -> list[Path]:
    symbol_dir = _symbol_dir(root, "mt5_bars", symbol)
    if symbol_dir is None:
        return []
    base = symbol_dir / f"timeframe={timeframe}"
    if not base.exists():
        return []
    return sorted([p for p in base.rglob("*") if p.suffix.lower() in (".parquet", ".csv")])


def find_collected_ticks(symbol: str, root: str | Path = "data/raw") -> list[Path]:
    symbol_dir = _symbol_dir(root, "mt5_ticks", symbol)
    if symbol_dir is None:
        return []
    return sorted([p for p in symbol_dir.rglob("*") if p.suffix.lower() in (".parquet", ".csv")])


def find_exness_ticks(symbol: str, root: str | Path = "data/raw") -> list[Path]:
    symbol_dir = _symbol_dir(root, "exness_ticks", symbol)
    if symbol_dir is None:
        return []
    return sorted([p for p in symbol_dir.rglob("*") if p.suffix.lower() in (".parquet", ".csv")])


def load_collected_bars(symbol: str, timeframe: str, root: str | Path = "data/raw") -> pd.DataFrame:
    files = find_collected_bars(symbol, timeframe, root)
    if not files:
        raise FileNotFoundError(f"no collected bars for {symbol} {timeframe} under {root}")
    frames = [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in files]
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates(subset=["time"] if "time" in frame.columns else None).sort_values("time").reset_index(drop=True)


def load_collected_ticks(symbol: str, root: str | Path = "data/raw") -> pd.DataFrame:
    files = find_collected_ticks(symbol, root)
    if not files:
        raise FileNotFoundError(f"no collected ticks for {symbol} under {root}")
    frames = [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in files]
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates(subset=["time_msc"] if "time_msc" in frame.columns else None).sort_values("time_msc").reset_index(drop=True)


def load_exness_ticks(symbol: str, root: str | Path = "data/raw") -> pd.DataFrame:
    """Load Exness archive ticks (stored under ``exness_ticks/``)."""
    files = find_exness_ticks(symbol, root)
    if not files:
        raise FileNotFoundError(f"no Exness ticks for {symbol} under {root}")
    frames = [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in files]
    frame = pd.concat(frames, ignore_index=True)
    if "time_msc" in frame.columns:
        frame = frame.drop_duplicates(subset=["time_msc"]).sort_values("time_msc")
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sample data generation (offline fallback)
# ---------------------------------------------------------------------------


def generate_sample_dataset(
    symbol: str,
    *,
    start: str = "2025-01-01",
    bar_periods: int = 20_000,
    tick_periods: int = 200_000,
    timeframe: str = "M1",
    out_dir: str | Path = SAMPLE_ROOT,
) -> dict[str, str]:
    """Generate deterministic sample bars + ticks so the full pipeline can run
    end-to-end on a machine without an MT5 terminal."""
    from slytrade.data.sample_generator import (
        generate_sample_bars,
        generate_sample_ticks,
        write_sample_frame,
    )
    from slytrade.data.timeframes import timeframe_duration

    out = Path(out_dir) / symbol
    out.mkdir(parents=True, exist_ok=True)
    bars_path = out / f"bars_{timeframe}.parquet"
    ticks_path = out / "ticks.parquet"
    write_sample_frame(
        generate_sample_bars(symbol=symbol, timeframe=timeframe, start=datetime.fromisoformat(start), periods=bar_periods),
        bars_path,
    )
    # Choose a tick cadence so ticks span (roughly) the same window as the bars,
    # giving the aligned dataset a realistic fresh-quote coverage ratio.
    bar_span_seconds = bar_periods * timeframe_duration(timeframe).total_seconds()
    tick_interval_ms = max(1000, int(bar_span_seconds * 1000 / tick_periods))
    write_sample_frame(
        generate_sample_ticks(
            symbol=symbol,
            start=datetime.fromisoformat(start),
            periods=tick_periods,
            tick_interval_ms=tick_interval_ms,
        ),
        ticks_path,
    )
    return {"bars": str(bars_path), "ticks": str(ticks_path)}


# ---------------------------------------------------------------------------
# Task: collect (bars for all timeframes + ticks)
# ---------------------------------------------------------------------------


def collect_all(
    symbol: str,
    *,
    lookback: str = "1y",
    timeframes: list[str] | None = None,
    include_ticks: bool = True,
    source: str = "hybrid",
    root: str | Path = "data/raw",
    sample_start: str = "2025-01-01",
) -> TaskResult:
    """Collect bars + ticks from their designated sources in one shot.

    SlyTrade's data model is fixed: **bars come from MT5, ticks come from the
    Exness archive**. ``source`` selects how strictly to follow it:

    * "hybrid" (default) — MT5 bars (every timeframe) + Exness archive ticks.
    * "auto" — try hybrid first; fall back gracefully if a source is down.
    * "mt5" — bars AND ticks from the MT5 terminal.
    * "exness" — Exness ticks resampled to bars (no terminal; tick-only fallback).
    * "samples" — deterministic synthetic data (offline smoke tests).
    """
    if timeframes is None:
        timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]

    if source == "hybrid":
        bars = _collect_bars_from_mt5(symbol, lookback=lookback, timeframes=timeframes, root=root)
        if not bars.ok:
            return TaskResult(False, f"MT5 bars failed (hybrid requires MT5): {bars.message}")
        if not include_ticks:
            return bars
        ticks = _download_exness_ticks(symbol, lookback=lookback, root=root)
        if not ticks.ok:
            return TaskResult(False, f"Exness ticks failed (hybrid requires Exness): {ticks.message}")
        return TaskResult(
            True,
            "hybrid collection complete (MT5 bars + Exness ticks)",
            {"source": "hybrid", "bars": bars.data or {}, "ticks": ticks.data or {}},
        )

    if source == "auto":
        try:
            bars = _collect_bars_from_mt5(symbol, lookback=lookback, timeframes=timeframes, root=root)
        except Exception as exc:  # pragma: no cover - broker dependent
            bars = TaskResult(False, str(exc))
        if bars.ok:
            if include_ticks:
                ticks = _download_exness_ticks(symbol, lookback=lookback, root=root)
                if ticks.ok:
                    return TaskResult(True, "hybrid collection complete (MT5 bars + Exness ticks)", {"source": "hybrid", "bars": bars.data or {}, "ticks": ticks.data or {}})
                console.print(f"[yellow]Exness ticks unavailable ({ticks.message}); falling back to MT5 ticks.[/yellow]")
                mt5_ticks = _collect_ticks_from_mt5(symbol, lookback=lookback, root=root)
                if mt5_ticks.ok:
                    return TaskResult(True, "collection complete (MT5 bars + MT5 ticks)", {"source": "mt5", "bars": bars.data or {}, "ticks": mt5_ticks.data or {}})
            return bars
        console.print(f"[yellow]MT5 bars unavailable ({bars.message}); falling back to Exness-only.[/yellow]")
        exness = _collect_from_exness(symbol, lookback=lookback, timeframes=timeframes)
        if exness.ok:
            return exness
        console.print(f"[yellow]Exness unavailable ({exness.message}); falling back to synthetic samples.[/yellow]")
        files = generate_sample_dataset(symbol, start=sample_start, out_dir=SAMPLE_ROOT)
        return TaskResult(True, "sample data generated", {"source": "samples", "files": files})

    if source == "mt5":
        try:
            return _collect_from_mt5(symbol, lookback=lookback, timeframes=timeframes, include_ticks=include_ticks, root=root)
        except Exception as exc:  # pragma: no cover - broker dependent
            return TaskResult(False, f"MT5 collection failed: {exc}")

    if source == "exness":
        return _collect_from_exness(symbol, lookback=lookback, timeframes=timeframes)

    files = generate_sample_dataset(symbol, start=sample_start, out_dir=SAMPLE_ROOT)
    if include_ticks:
        console.print(f"[green]Generated sample bars + ticks for {symbol} (synthetic).[/green]")
    else:
        console.print(f"[green]Generated sample bars for {symbol} (synthetic).[/green]")
    return TaskResult(True, "sample data generated", {"source": "samples", "files": files})


def _collect_bars_from_mt5(
    symbol: str,
    *,
    lookback: str,
    timeframes: list[str],
    root: str | Path,
) -> TaskResult:
    """Collect MT5 bars for every timeframe (bars are always an MT5 source)."""
    from slytrade.data.mt5_collectors import MT5BarCollector
    from slytrade.data.storage import MarketDataStorage
    from slytrade.data.time import date_range_from_lookback

    mt5 = _load_mt5()
    _init_mt5(mt5)
    storage = MarketDataStorage(Path(root))
    start_dt, end_dt = date_range_from_lookback(lookback)
    try:
        summary: dict[str, Any] = {}
        for timeframe in timeframes:
            result = MT5BarCollector(mt5, storage).collect(symbol, timeframe, start_dt, end_dt, chunk_size="month")
            summary[f"bars_{timeframe}"] = result.rows
            console.print(f"  bars {timeframe}: {result.rows} rows in {result.file_count} files")
        return TaskResult(True, "MT5 bars collected", {"source": "mt5", **summary})
    finally:
        _shutdown_mt5(mt5)


def _collect_ticks_from_mt5(
    symbol: str,
    *,
    lookback: str,
    root: str | Path,
) -> TaskResult:
    """Collect MT5 ticks (fallback when the Exness archive is unreachable)."""
    from slytrade.data.mt5_collectors import MT5TickCollector
    from slytrade.data.storage import MarketDataStorage
    from slytrade.data.time import date_range_from_lookback

    mt5 = _load_mt5()
    _init_mt5(mt5)
    storage = MarketDataStorage(Path(root))
    start_dt, end_dt = date_range_from_lookback(lookback)
    try:
        result = MT5TickCollector(mt5, storage).collect(symbol, start_dt, end_dt, chunk_size="day")
        console.print(f"  ticks: {result.rows} rows in {result.file_count} files")
        return TaskResult(True, "MT5 ticks collected", {"source": "mt5", "ticks": result.rows})
    finally:
        _shutdown_mt5(mt5)


def _download_exness_ticks(
    symbol: str,
    *,
    lookback: str,
    root: str | Path = "data/raw",
) -> TaskResult:
    """Download Exness archive ticks (ticks are always an Exness source)."""
    from slytrade.data.exness_archive import ExnessArchiveDownloader, normalize_exness_symbol
    from slytrade.data.time import date_range_from_lookback

    archive_symbol = normalize_exness_symbol(symbol)
    start_dt, end_dt = date_range_from_lookback(lookback)
    downloader = ExnessArchiveDownloader(str(root))
    result = downloader.collect(archive_symbol, start_dt, end_dt, continue_on_error=True)
    if result.rows <= 0:
        return TaskResult(False, "Exness archive returned no tick rows")
    console.print(f"  ticks: {result.rows} rows (Exness archive) in {result.file_count} files")
    return TaskResult(True, "Exness ticks downloaded", {"source": "exness", "ticks": result.rows})


def _collect_from_exness(symbol: str, *, lookback: str, timeframes: list[str]) -> TaskResult:
    """Exness-only fallback: download ticks and resample them to bars."""
    from slytrade.data.exness_archive import normalize_exness_symbol
    from slytrade.data.resample import resample_ticks_to_bars
    from slytrade.data.sample_generator import write_sample_frame

    archive_symbol = normalize_exness_symbol(symbol)
    out_root = Path(EXNESS_DERIVED_ROOT) / archive_symbol
    downloaded = _download_exness_ticks(symbol, lookback=lookback)
    if not downloaded.ok:
        return downloaded
    try:
        ticks = load_exness_ticks(archive_symbol)
        if ticks.empty:
            return TaskResult(False, "Exness archive returned no ticks")
    except Exception as exc:  # pragma: no cover - network dependent
        return TaskResult(False, f"Exness collection failed: {exc}")

    files: dict[str, str] = {}
    for timeframe in timeframes:
        bars = resample_ticks_to_bars(ticks, timeframe, symbol=archive_symbol)
        if bars.empty:
            continue
        out_root.mkdir(parents=True, exist_ok=True)
        bars_path = out_root / f"bars_{timeframe}.parquet"
        write_sample_frame(bars, bars_path)
        files[f"bars_{timeframe}"] = str(bars_path)
        console.print(f"  bars {timeframe}: {len(bars)} bars (resampled from Exness ticks)")
    ticks_path = out_root / "ticks.parquet"
    write_sample_frame(ticks, ticks_path)
    files["ticks"] = str(ticks_path)
    console.print(f"  ticks: {len(ticks)} (Exness archive)")
    return TaskResult(True, "Exness collection complete (ticks resampled to bars)", {"source": "exness", "files": files})


def _collect_from_mt5(
    symbol: str,
    *,
    lookback: str,
    timeframes: list[str],
    include_ticks: bool,
    root: str | Path,
) -> TaskResult:
    """MT5-only path: bars for every timeframe plus optional MT5 ticks."""
    bars = _collect_bars_from_mt5(symbol, lookback=lookback, timeframes=timeframes, root=root)
    if not bars.ok:
        return bars
    if include_ticks:
        ticks = _collect_ticks_from_mt5(symbol, lookback=lookback, root=root)
        if ticks.ok:
            return TaskResult(True, "collection complete (MT5 bars + MT5 ticks)", {"source": "mt5", "bars": bars.data or {}, "ticks": ticks.data or {}})
    return bars


def _load_mt5() -> Any:
    try:
        import MetaTrader5 as mt5

        return mt5
    except ImportError:
        from mt5linux import MetaTrader5 as MT5Linux

        return MT5Linux()


def _init_mt5(mt5: Any) -> None:
    if hasattr(mt5, "initialize"):
        ok = mt5.initialize()
        if ok is False:
            raise RuntimeError("MT5 initialize() returned False")


def _shutdown_mt5(mt5: Any) -> None:
    if hasattr(mt5, "shutdown"):
        mt5.shutdown()


# ---------------------------------------------------------------------------
# Task: align
# ---------------------------------------------------------------------------


def align(
    symbol: str,
    *,
    bars_file: str | None = None,
    ticks_file: str | None = None,
    timeframe: str = "M1",
    root: str | Path = "data/raw",
    out_dir: str | Path | None = None,
) -> TaskResult:
    from slytrade.data.alignment import align_market_data, render_manifest, save_aligned_dataset

    # Resolve inputs: explicit files, or the designated sources (MT5 bars +
    # Exness ticks), or sample data as a last resort.
    bars = None
    ticks = None
    bars_source = "sample_bars"
    ticks_source = "sample_ticks"

    if bars_file is not None:
        bars = _load_frame(bars_file)
    else:
        for label, path in (
            ("mt5", None),
            ("exness_derived", Path(EXNESS_DERIVED_ROOT) / symbol / f"bars_{timeframe}.parquet"),
            ("sample", Path(SAMPLE_ROOT) / symbol / f"bars_{timeframe}.parquet"),
        ):
            if label == "mt5":
                try:
                    bars = load_collected_bars(symbol, timeframe, root)
                except FileNotFoundError:
                    continue
            else:
                assert path is not None
                if not path.exists():
                    continue
                bars = _load_frame(path)
                console.print(f"[yellow]Using {path}[/yellow]")
            bars_source = {"mt5": "mt5_bars", "exness_derived": "exness_derived", "sample": "sample_bars"}[label]
            break
        if bars is None:
            return TaskResult(False, f"no bars found for {symbol} {timeframe}; run collection first")

    if ticks_file is not None:
        ticks = _load_frame(ticks_file)
    else:
        # Ticks prefer the Exness archive (designated tick source), then MT5
        # ticks, then the derived/sample tick files.
        for label, path in (
            ("exness", None),
            ("mt5", None),
            ("exness_derived", Path(EXNESS_DERIVED_ROOT) / symbol / "ticks.parquet"),
            ("sample", Path(SAMPLE_ROOT) / symbol / "ticks.parquet"),
        ):
            if label == "exness":
                try:
                    ticks = load_exness_ticks(symbol, root)
                except FileNotFoundError:
                    continue
            elif label == "mt5":
                try:
                    ticks = load_collected_ticks(symbol, root)
                except FileNotFoundError:
                    continue
            else:
                assert path is not None
                if not path.exists():
                    continue
                ticks = _load_frame(path)
                console.print(f"[yellow]Using {path}[/yellow]")
            ticks_source = {"exness": "exness_ticks", "mt5": "mt5_ticks", "exness_derived": "exness_ticks", "sample": "sample_ticks"}[label]
            break
        if ticks is None:
            return TaskResult(False, f"no ticks found for {symbol}; run collection first")

    output = Path(out_dir) if out_dir else Path("data/processed/aligned") / symbol
    dataset = align_market_data(
        bars,
        ticks,
        timeframe=timeframe,
        canonical_symbol=symbol,
        bar_source=bars_source,
        tick_source=ticks_source,
        include_ict_features=True,
        include_tick_features=True,
        require_fresh_quotes=False,
    )
    manifest = save_aligned_dataset(dataset, output, source_bars_file=bars_file, source_ticks_file=ticks_file, copy_ticks=False)
    render_manifest(manifest, console=console)
    return TaskResult(True, f"aligned dataset ready at {output}", {"bars_file": manifest.files["bars"], "rows": len(dataset.bars)})


def _load_frame(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Task: backtest (persona-adaptive, managed exits)
# ---------------------------------------------------------------------------


def default_point_value(symbol: str) -> float:
    """Conservative per-symbol point value for sizing.

    For gold/silver a 1.0 price move at 1.0 lot is ~$100; for FX pairs it is
    ~$1 per 1.0 lot per point. Real broker specs (from the symbol spec file)
    always take precedence in the live paths.
    """
    normalized = symbol.upper()
    if normalized in ("XAUUSD", "XAGUSD", "XAUUSDm", "XAGUSDm"):
        return 100.0
    return 1.0


def backtest(
    bars_file: str,
    *,
    strategy: str = "persona-adaptive",
    symbol: str | None = None,
    initial_balance: float = 100_000.0,
    point_size: float = 0.01,
    point_value: float | None = None,
    commission: float = 0.0,
) -> TaskResult:
    from slytrade.backtest.engine import BacktestConfig
    from slytrade.backtest.reporting import (
        infer_symbol,
        load_bars_file,
        render_backtest_report,
        run_managed_aligned_backtest_from_bars,
    )

    bars = load_bars_file(Path(bars_file))
    resolved = infer_symbol(bars, symbol)
    point_value = point_value if point_value is not None else default_point_value(resolved)
    result = run_managed_aligned_backtest_from_bars(
        bars,
        strategy_name=strategy,
        symbol=symbol,
        volume=0.1,
        point_value=point_value,
        config=BacktestConfig(
            initial_balance=initial_balance,
            point_size=point_size,
            point_value=point_value,
            commission_per_volume=commission,
        ),
    )
    render_backtest_report(result, strategy_name=strategy, console=console)
    return TaskResult(
        True,
        "backtest complete",
        {
            "total_return": result.metrics.total_return,
            "max_drawdown": result.metrics.max_drawdown,
            "sharpe_like": result.metrics.sharpe_like,
            "trades": result.metrics.trades,
            "final_equity": result.metrics.final_equity,
        },
    )


# ---------------------------------------------------------------------------
# Task: train (+ artifact + registry)
# ---------------------------------------------------------------------------


def train(
    bars_file: str,
    *,
    symbol: str | None = None,
    algorithm: str = "ppo",
    total_timesteps: int = 50_000,
    seed: int = 42,
    policy: str = "mlp",
    reward: str = "risk_adjusted",
    artifacts_dir: str | Path = "models/artifacts",
    registry_path: str | Path = "models/registry.jsonl",
) -> TaskResult:
    from slytrade.backtest.reporting import infer_symbol, load_bars_file
    from slytrade.rl.dataset import build_rl_dataset
    from slytrade.rl.deployment import save_model_artifact
    from slytrade.rl.tracking import maybe_end_run, maybe_log_metrics, maybe_log_params, maybe_start_run
    from slytrade.rl.walkforward import evaluate_policy, resolve_algorithm, train_policy, train_ppo

    algorithm = resolve_algorithm(algorithm)
    bars = load_bars_file(Path(bars_file))
    resolved = infer_symbol(bars, symbol)
    bars = bars[bars["symbol"] == resolved].copy()
    try:
        dataset = build_rl_dataset(bars)
        scaler_params = dataset.fit_scaler(0, len(dataset.bars))
        env = dataset.env_factory(0, len(dataset.bars), seed=seed, scaler_params=scaler_params)
    except ImportError as exc:
        return TaskResult(False, f"RL dependencies not installed: {exc}")
    env.config = _with_reward(env.config, reward)

    run = maybe_start_run("slytrade-rl", run_name=f"{algorithm}-{resolved}-{seed}")
    maybe_log_params(run, {"algorithm": algorithm, "symbol": resolved, "seed": seed, "timesteps": total_timesteps, "policy": policy, "reward": reward})
    try:
        if algorithm == "ppo":
            model = train_ppo(env, total_timesteps=total_timesteps, seed=seed, policy_type=policy, model_dir=str(Path(artifacts_dir) / f"{algorithm}-{resolved}-{seed}"))
        else:
            model = train_policy(algorithm, env, total_timesteps=total_timesteps, seed=seed, policy_type=policy)
        results = evaluate_policy(model, env, episodes=3, seed=seed)
        maybe_log_metrics(run, {f"mean_{key}": float(value) for key, value in results.items() if isinstance(value, (int, float))})
    finally:
        maybe_end_run(run)

    record = save_model_artifact(
        model,
        model_id=f"{algorithm}-{resolved}-{seed}",
        algorithm=algorithm,
        symbol=resolved,
        feature_columns=list(dataset.features.columns),
        scaler_params=scaler_params,
        env_config={
            "reward_type": reward,
            "policy": policy,
            "seed": seed,
            "total_timesteps": total_timesteps,
        },
        metrics={f"mean_{key}": float(value) for key, value in results.items() if isinstance(value, (int, float))},
        artifacts_dir=artifacts_dir,
        registry_path=registry_path,
    )
    console.print(f"[green]Trained {algorithm.upper()} ({policy}) policy; artifact registered as {record['model_id']}[/green]")
    for key, value in results.items():
        console.print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    return TaskResult(True, "training complete", {"model_id": record["model_id"], "metrics": {k: float(v) for k, v in results.items() if isinstance(v, (int, float))}})


def _with_reward(config, reward: str):
    if reward in ("risk_adjusted", "raw"):
        from dataclasses import replace

        return replace(config, reward_type=reward)
    return config


# ---------------------------------------------------------------------------
# Task: walk-forward
# ---------------------------------------------------------------------------


def walk_forward(
    bars_file: str,
    *,
    symbol: str | None = None,
    total_timesteps: int = 10_000,
    seed: int = 42,
    train_window: int = 10_000,
    validation_window: int = 2_000,
    test_window: int = 2_000,
    embargo: int = 100,
) -> TaskResult:
    from slytrade.backtest.reporting import infer_symbol, load_bars_file
    from slytrade.rl.dataset import build_rl_dataset
    from slytrade.rl.walkforward import make_walk_forward_folds, walk_forward_validation

    bars = load_bars_file(Path(bars_file))
    resolved = infer_symbol(bars, symbol)
    bars = bars[bars["symbol"] == resolved].copy()
    try:
        dataset = build_rl_dataset(bars)
        folds = make_walk_forward_folds(
            len(dataset.bars),
            train_window=train_window,
            validation_window=validation_window,
            test_window=test_window,
            embargo=embargo,
        )
        table = walk_forward_validation(dataset, folds, total_timesteps=total_timesteps, seed=seed)
    except ImportError as exc:
        return TaskResult(False, f"RL dependencies not installed: {exc}")
    console.print(table.to_string(index=False))
    return TaskResult(True, "walk-forward validation complete", {"folds": len(folds)})


# ---------------------------------------------------------------------------
# Task: promote
# ---------------------------------------------------------------------------


def promote(model_id: str, *, stage: str = "paper", registry_path: str | Path = "models/registry.jsonl") -> TaskResult:
    from slytrade.rl.deployment import promote_artifact

    record = promote_artifact(model_id, stage=stage, registry_path=registry_path)
    console.print(f"[green]Promoted {model_id} to stage '{stage}'.[/green]")
    return TaskResult(True, f"promoted {model_id} -> {stage}", {"record": record})


# ---------------------------------------------------------------------------
# Task: full pipeline from scratch
# ---------------------------------------------------------------------------


def full_pipeline(
    symbol: str,
    *,
    lookback: str = "1y",
    source: str = "auto",
    algorithm: str = "ppo",
    total_timesteps: int = 50_000,
    policy: str = "mlp",
    reward: str = "risk_adjusted",
    promote_stage: str = "paper",
) -> TaskResult:
    """Run collection → alignment → backtest → train → walk-forward → promote."""
    steps: list[str] = []
    collected = collect_all(symbol, lookback=lookback, source=source)
    if not collected.ok:
        return collected
    steps.append("collect")

    aligned = align(symbol)
    if not aligned.ok:
        return aligned
    bars_file = aligned.data["bars_file"] if aligned.data else None
    if not bars_file:
        return TaskResult(False, "alignment did not produce a bars file")
    steps.append("align")

    backtest(bars_file, strategy="persona-adaptive", symbol=symbol)
    steps.append("backtest")

    trained = train(bars_file, symbol=symbol, algorithm=algorithm, total_timesteps=total_timesteps, policy=policy, reward=reward)
    if not trained.ok:
        return trained
    steps.append("train")

    walk_forward(bars_file, symbol=symbol)
    steps.append("walk_forward")

    model_id = trained.data["model_id"] if trained.data else None
    if model_id:
        promote(model_id, stage=promote_stage)
        steps.append("promote")

    console.print(f"[bold green]Full pipeline complete: {' → '.join(steps)}[/bold green]")
    return TaskResult(True, "full pipeline complete", {"steps": steps, "model_id": model_id})
