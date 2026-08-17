"""High-level, task-oriented operations for the full trading pipeline.

These functions are the engine behind both the CLI commands and the Rich GUI.
Each task does the whole job (collect, align, backtest, train, walk-forward,
promote, paper, demo) with sensible defaults so a user never has to assemble
flags by hand. They return small result dicts the UI can render.
"""

from __future__ import annotations

import getpass
import importlib.util
import os
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


def find_merged_ticks(symbol: str, root: str | Path = "data/raw") -> list[Path]:
    symbol_dir = _symbol_dir(root, "merged_ticks", symbol)
    if symbol_dir is None:
        return []
    return sorted([p for p in symbol_dir.rglob("*") if p.suffix.lower() in (".parquet", ".csv")])


def load_merged_ticks(symbol: str, root: str | Path = "data/raw") -> pd.DataFrame:
    """Load merged ticks (Exness history + MT5 recent), the preferred tick set."""
    files = find_merged_ticks(symbol, root)
    if not files:
        raise FileNotFoundError(f"no merged ticks for {symbol} under {root}")
    frames = [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in files]
    frame = pd.concat(frames, ignore_index=True)
    if "time_msc" in frame.columns:
        frame = frame.drop_duplicates(subset=["time_msc"]).sort_values("time_msc")
    return frame.reset_index(drop=True)


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
    clean: bool = False,
) -> TaskResult:
    """Collect bars + ticks from their designated sources in one shot.

    SlyTrade's data model: **bars come from MT5, ticks come from the Exness
    archive**, with the recent few days of ticks additionally collected from
    MT5 and merged in so the freshest bars never have a stale quote (the Exness
    archive lags by roughly a day). ``source`` selects how strictly to follow it:

    * "hybrid" (default) — MT5 bars (every timeframe) + merged ticks
      (Exness history + MT5 recent).
    * "auto" — try hybrid first; fall back gracefully if a source is down.
    * "mt5" — bars AND ticks from the MT5 terminal.
    * "exness" — Exness ticks resampled to bars (no terminal; tick-only fallback).
    * "samples" — deterministic synthetic data (offline smoke tests).
    """
    if timeframes is None:
        timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]

    from slytrade.progress import stage

    stage(f"Collect {symbol} ({lookback}) — bars from MT5, ticks from Exness")
    if clean:
        cleaned = clean_all()
        if not cleaned.ok:
            return cleaned

    # Recreate any deleted data directories, then verify the path this source
    # actually writes to is writable. The `samples` source writes to SAMPLE_ROOT,
    # not to `root`, so it must not require data/raw to exist.
    if source == "samples":
        writable_error = _ensure_writable_root(SAMPLE_ROOT)
    else:
        writable_error = _ensure_standard_dirs() or _ensure_writable_root(root)
    if writable_error is not None:
        return writable_error

    if source == "hybrid":
        bars = _collect_bars_from_mt5(symbol, lookback=lookback, timeframes=timeframes, root=root)
        if not bars.ok:
            return TaskResult(False, f"MT5 bars failed (hybrid requires MT5): {bars.message}")
        if not include_ticks:
            return bars
        ticks = _merge_tick_sources(symbol, lookback=lookback, root=root)
        if not ticks.ok:
            return TaskResult(False, f"tick collection failed (hybrid requires ticks): {ticks.message}")
        return TaskResult(
            True,
            "hybrid collection complete (MT5 bars + merged Exness/MT5 ticks)",
            {"source": "hybrid", "bars": bars.data or {}, "ticks": ticks.data or {}},
        )

    if source == "auto":
        try:
            bars = _collect_bars_from_mt5(symbol, lookback=lookback, timeframes=timeframes, root=root)
        except Exception as exc:  # pragma: no cover - broker dependent
            bars = TaskResult(False, str(exc))
        if bars.ok:
            if include_ticks:
                ticks = _merge_tick_sources(symbol, lookback=lookback, root=root)
                if ticks.ok:
                    return TaskResult(True, "hybrid collection complete (MT5 bars + merged Exness/MT5 ticks)", {"source": "hybrid", "bars": bars.data or {}, "ticks": ticks.data or {}})
                console.print(f"[yellow]Merged ticks unavailable ({ticks.message}); falling back to Exness-only ticks.[/yellow]")
                exness = _download_exness_ticks(symbol, lookback=lookback, root=root)
                if exness.ok:
                    return TaskResult(True, "collection complete (MT5 bars + Exness ticks)", {"source": "hybrid", "bars": bars.data or {}, "ticks": exness.data or {}})
                console.print(f"[yellow]Exness unavailable ({exness.message}); falling back to MT5 ticks.[/yellow]")
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


def _nearest_existing_owner(path: Path) -> tuple[Path, str]:
    """Find the nearest existing ancestor of `path` and who owns it."""
    ancestor = path
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    owner = "unknown"
    try:
        stat = ancestor.stat()
        try:
            import pwd

            owner = f"{pwd.getpwuid(stat.st_uid).pw_name} (uid {stat.st_uid})"
        except (ImportError, KeyError):  # pragma: no cover - uid without passwd entry
            owner = f"uid {stat.st_uid}"
    except OSError:  # pragma: no cover
        pass
    return ancestor, owner


def _ensure_writable_root(root: str | Path) -> TaskResult | None:
    """Create (or re-create) a data path and verify it is writable.

    The path is created if missing — so a folder the operator deleted is
    recreated automatically. A friendly, self-diagnosing error is returned only
    when creation genuinely fails (the usual causes: a parent owned by root
    from sudo/Docker, or an external drive with a filesystem that ignores Unix
    ownership such as exFAT/NTFS).
    """
    root_path = Path(root)
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        probe = root_path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except PermissionError:
        ancestor, owner = _nearest_existing_owner(root_path)
        try:
            current = f"{getpass.getuser()} (uid {os.getuid()})"
        except Exception:  # pragma: no cover
            current = f"uid {os.getuid()}"
        return TaskResult(
            False,
            f"cannot create or write to '{root_path}' (permission denied).\n"
            f"  you are running as: {current}\n"
            f"  nearest existing path '{ancestor}' is owned by: {owner}\n"
            "If that path is owned by root (Docker creates host bind dirs as root), fix with:\n"
            f"  sudo chown -R \"$USER\":\"$USER\" \"{root_path}\"\n"
            "If chown reports 'Operation not permitted', the drive is exFAT/NTFS and "
            "ignores Unix ownership. Easiest fix: move the project to your home drive:\n"
            f"  mv \"{Path.cwd()}\" ~/\n"
            "or re-mount the external drive with your uid/gid.",
        )
    except OSError as exc:  # pragma: no cover - filesystem dependent
        return TaskResult(False, f"cannot create or write to '{root_path}': {exc}")
    return None


def _ensure_standard_dirs() -> TaskResult | None:
    """Recreate the standard data/output directory tree if any of it is missing.

    Called at the start of the pipeline tasks so a deleted ``data/``, ``models/``,
    ``logs/`` or ``state/`` folder is restored without manual intervention.
    """
    for directory in (
        Path("data") / "raw",
        Path("data") / "processed",
        Path("data") / "exness_derived",
        Path("data") / "samples",
        Path("models"),
        Path("logs"),
        Path("state"),
    ):
        error = _ensure_writable_root(directory)
        if error is not None:
            return error
    return None


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

    try:
        mt5 = _load_mt5()
        _init_mt5(mt5)
    except Exception as exc:  # pragma: no cover - broker dependent
        return TaskResult(False, f"could not connect to MT5: {exc}")
    storage = MarketDataStorage(Path(root))
    start_dt, end_dt = date_range_from_lookback(lookback)
    try:
        summary: dict[str, Any] = {}
        for timeframe in timeframes:
            result = MT5BarCollector(mt5, storage).collect(symbol, timeframe, start_dt, end_dt, chunk_size="month")
            summary[f"bars_{timeframe}"] = result.rows
            console.print(f"  bars {timeframe}: {result.rows} rows in {result.file_count} files")
        return TaskResult(True, "MT5 bars collected", {"source": "mt5", **summary})
    except PermissionError:
        return TaskResult(False, f"permission denied writing bars under {root}; see the data-root check above")
    except Exception as exc:  # pragma: no cover - broker dependent
        return TaskResult(False, f"MT5 bar collection failed: {exc}")
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

    try:
        mt5 = _load_mt5()
        _init_mt5(mt5)
    except Exception as exc:  # pragma: no cover - broker dependent
        return TaskResult(False, f"could not connect to MT5: {exc}")
    storage = MarketDataStorage(Path(root))
    start_dt, end_dt = date_range_from_lookback(lookback)
    try:
        result = MT5TickCollector(mt5, storage).collect(symbol, start_dt, end_dt, chunk_size="day")
        console.print(f"  ticks: {result.rows} rows in {result.file_count} files")
        return TaskResult(True, "MT5 ticks collected", {"source": "mt5", "ticks": result.rows})
    except PermissionError:
        return TaskResult(False, f"permission denied writing ticks under {root}; see the data-root check above")
    except Exception as exc:  # pragma: no cover - broker dependent
        return TaskResult(False, f"MT5 tick collection failed: {exc}")
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
    try:
        downloader = ExnessArchiveDownloader(str(root))
        result = downloader.collect(archive_symbol, start_dt, end_dt, continue_on_error=True)
    except PermissionError:
        return TaskResult(False, f"permission denied writing Exness ticks under {root}; see the data-root check above")
    except Exception as exc:  # pragma: no cover - network dependent
        return TaskResult(False, f"Exness download failed: {exc}")
    if result.rows <= 0:
        return TaskResult(False, "Exness archive returned no tick rows")
    console.print(f"  ticks: {result.rows} rows (Exness archive) in {result.file_count} files")
    return TaskResult(True, "Exness ticks downloaded", {"source": "exness", "ticks": result.rows})


def _download_exness_ticks_before(
    symbol: str,
    *,
    lookback: str,
    root: str | Path = "data/raw",
    end_override: pd.Timestamp,
) -> TaskResult:
    """Download Exness ticks only up to ``end_override`` (MT5 coverage start).

    MT5 already covers [end_override, now], so downloading the Exness archive
    beyond that point wastes bandwidth and CPU on data that will be discarded.
    """
    from slytrade.data.exness_archive import ExnessArchiveDownloader, normalize_exness_symbol
    from slytrade.data.time import date_range_from_lookback

    archive_symbol = normalize_exness_symbol(symbol)
    start_dt, _ = date_range_from_lookback(lookback)
    end_dt = end_override.to_pydatetime()
    if end_dt <= start_dt:
        console.print("  Exness: MT5 already covers the whole lookback; skipping archive download.")
        return TaskResult(True, "Exness not needed (MT5 covers the full lookback)", {"source": "exness", "ticks": 0})
    try:
        downloader = ExnessArchiveDownloader(str(root))
        result = downloader.collect(archive_symbol, start_dt, end_dt, continue_on_error=True)
    except PermissionError:
        return TaskResult(False, f"permission denied writing Exness ticks under {root}; see the data-root check above")
    except Exception as exc:  # pragma: no cover - network dependent
        return TaskResult(False, f"Exness download failed: {exc}")
    if result.rows <= 0:
        return TaskResult(False, "Exness archive returned no tick rows")
    console.print(f"  ticks: {result.rows} rows (Exness archive, before MT5 coverage)")
    return TaskResult(True, "Exness ticks downloaded", {"source": "exness", "ticks": result.rows})


# Recent MT5 tick window (days) merged on top of the Exness history so the
# freshest bars never carry a stale quote from the archive's ~1-day lag.
RECENT_MT5_TICK_DAYS = 3


def _merge_tick_sources(
    symbol: str,
    *,
    lookback: str,
    root: str | Path = "data/raw",
    recent_days: int = RECENT_MT5_TICK_DAYS,
) -> TaskResult:
    """Merge MT5 ticks (authoritative) with Exness ticks (older history).

    MT5 is collected first for the FULL lookback (it covers from "now" back to
    the start of its tick history). The Exness archive is then downloaded only
    for the period BEFORE MT5's coverage begins, so there are no gaps and no
    stale end-of-window bars from the archive's ~1-day lag. Both sides are
    merged month-by-month in a single streaming pass (memory O(one month)).
    """
    from slytrade.data.exness_archive import normalize_exness_symbol
    from slytrade.data.tick_stream import merge_mt5_exness_streaming, min_tick_time_ns

    canonical = normalize_exness_symbol(symbol)

    # 1) MT5 ticks — full lookback (authoritative for the recent part).
    mt5_res = _collect_ticks_from_mt5(symbol, lookback=lookback, root=root)
    mt5_files = find_collected_ticks(symbol, root=root) if mt5_res.ok else []
    if mt5_files:
        console.print(f"  MT5 ticks: {len(mt5_files)} files (full lookback, authoritative)")
    else:  # pragma: no cover - broker dependent
        console.print(f"[yellow]MT5 ticks unavailable ({mt5_res.message if not mt5_res.ok else 'no files'}); using Exness only.[/yellow]")

    # 2) Exness archive — only for the period BEFORE MT5 coverage.
    if mt5_files:
        mt5_start_ns = min_tick_time_ns(mt5_files)
        if mt5_start_ns is not None:
            exness = _download_exness_ticks_before(symbol, lookback=lookback, root=root, end_override=pd.Timestamp(mt5_start_ns, tz="UTC"))
            if not exness.ok and exness.data is None:
                # A hard failure (network/permissions) aborts; "not needed"
                # returns ok with ticks=0 and is fine.
                return exness
        else:  # pragma: no cover
            exness = TaskResult(False, "MT5 tick files are empty")
    else:
        exness = _download_exness_ticks(symbol, lookback=lookback, root=root)
        if not exness.ok:
            return exness

    exness_files = find_exness_ticks(canonical, root=root)
    if not mt5_files and not exness_files:
        return TaskResult(False, "no tick data available (both MT5 and Exness empty)")

    try:
        total_rows = merge_mt5_exness_streaming(
            exness_files,
            mt5_files,
            out_root=Path(root) / "merged_ticks",
            symbol=canonical,
        )
    except Exception as exc:  # pragma: no cover - io dependent
        return TaskResult(False, f"tick merge failed: {exc}")

    console.print(f"  merged ticks: {total_rows} rows -> {Path(root) / 'merged_ticks'}")
    return TaskResult(True, "merged ticks written", {"source": "merged", "ticks": total_rows})


def _write_merged_ticks(symbol: str, ticks: pd.DataFrame, *, root: str | Path = "data/raw") -> None:
    """Write a merged tick frame, chunked by month under ``merged_ticks/``."""
    out = Path(root) / "merged_ticks" / f"symbol={symbol}"
    if "time_msc" not in ticks.columns or ticks.empty:
        out.mkdir(parents=True, exist_ok=True)
        ticks.to_parquet(out / "ticks.parquet", index=False)
        return
    ticks = ticks.copy()
    ticks["time_msc"] = pd.to_datetime(ticks["time_msc"], utc=True)
    for (year, month), group in ticks.groupby([ticks["time_msc"].dt.year, ticks["time_msc"].dt.month]):
        directory = out / f"year={year}" / f"month={int(month):02d}"
        directory.mkdir(parents=True, exist_ok=True)
        group.to_parquet(directory / "ticks.parquet", index=False)


def _collect_from_exness(symbol: str, *, lookback: str, timeframes: list[str]) -> TaskResult:
    """Exness-only fallback: download ticks and resample them to bars."""
    from slytrade.data.exness_archive import normalize_exness_symbol
    from slytrade.data.sample_generator import write_sample_frame
    from slytrade.data.tick_stream import resample_ticks_to_bars_streaming

    archive_symbol = normalize_exness_symbol(symbol)
    out_root = Path(EXNESS_DERIVED_ROOT) / archive_symbol
    downloaded = _download_exness_ticks(symbol, lookback=lookback)
    if not downloaded.ok:
        return downloaded
    tick_files = find_exness_ticks(archive_symbol)
    if not tick_files:
        return TaskResult(False, "Exness ticks downloaded but could not be found")

    files: dict[str, str] = {}
    for timeframe in timeframes:
        bars = resample_ticks_to_bars_streaming(tick_files, timeframe, symbol=archive_symbol)
        if bars.empty:
            continue
        out_root.mkdir(parents=True, exist_ok=True)
        bars_path = out_root / f"bars_{timeframe}.parquet"
        write_sample_frame(bars, bars_path)
        files[f"bars_{timeframe}"] = str(bars_path)
        console.print(f"  bars {timeframe}: {len(bars)} bars (resampled from Exness ticks)")
    # Reference the raw Exness ticks rather than copying the full set.
    ticks_path = tick_files[0]
    files["ticks"] = str(ticks_path)
    console.print(f"  ticks: {len(tick_files)} files (Exness archive, streaming)")
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


def _tick_symbol_dir(files: list[Path]) -> str:
    """Return the symbol=… ancestor directory of the first tick file."""
    directory = files[0].parent
    while directory.name and not directory.name.startswith("symbol="):
        directory = directory.parent
    return str(directory)


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
    from slytrade.data.tick_stream import align_market_data_streaming
    from slytrade.progress import stage

    stage(f"Align {symbol} {timeframe} — ticks → features → decision quotes → MTF")
    output = Path(out_dir) if out_dir else Path("data/processed/aligned") / symbol
    # Recreate any deleted data directories and ensure the output path is
    # writable (self-heals a deleted data/processed tree).
    writable_error = _ensure_writable_root(output.parent)
    if writable_error is not None:
        return writable_error

    # Resolve inputs: explicit files, or the designated sources (MT5 bars +
    # Exness ticks), or sample data as a last resort.
    bars = None
    bars_source = "sample_bars"

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

    # Tick discovery returns either partitioned FILES (streaming, memory-bounded)
    # or a single explicit/sample file (in-memory, small).
    tick_files: list[Path] | None = None
    ticks = None
    ticks_source = "sample_ticks"
    tick_file_dir: str | None = None

    if ticks_file is not None:
        ticks = _load_frame(ticks_file)
    else:
        for label, finder in (
            ("merged", find_merged_ticks),
            ("exness", find_exness_ticks),
            ("mt5", find_collected_ticks),
        ):
            found = finder(symbol, root)
            if found:
                tick_files = found
                ticks_source = {"merged": "merged_ticks", "exness": "exness_ticks", "mt5": "mt5_ticks"}[label]
                tick_file_dir = _tick_symbol_dir(found)
                console.print(f"[green]Streaming {len(found)} tick files ({label})[/green]")
                break
        if tick_files is None:
            for label, path in (
                ("exness_derived", Path(EXNESS_DERIVED_ROOT) / symbol / "ticks.parquet"),
                ("sample", Path(SAMPLE_ROOT) / symbol / "ticks.parquet"),
            ):
                if path.exists():
                    ticks = _load_frame(path)
                    ticks_source = {"exness_derived": "exness_ticks", "sample": "sample_ticks"}[label]
                    console.print(f"[yellow]Using {path}[/yellow]")
                    break
            if ticks is None and tick_files is None:
                return TaskResult(False, f"no ticks found for {symbol}; run collection first")

    if tick_files is not None:
        try:
            dataset = align_market_data_streaming(
                bars,
                tick_files,
                timeframe=timeframe,
                canonical_symbol=symbol,
                bar_source=bars_source,
                tick_source=ticks_source,
                max_quote_age_seconds=5.0,
                min_fresh_coverage=0.95,
                include_ict_features=True,
                # Drop bars without a fresh decision quote so the delivered
                # dataset has zero stale bars/ticks (edge-of-history gaps from
                # the archive are trimmed, never trained on).
                require_fresh_quotes=True,
            )
            source_ticks_file = tick_file_dir
        except Exception as exc:
            return TaskResult(False, f"streaming alignment failed: {exc}")
    else:
        assert ticks is not None
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
        source_ticks_file = ticks_file

    # Inject the higher-timeframe (MTF) features from the collected M5/M15/
    # H1/H4/D1 bars. This is the "always watching every timeframe" layer: the
    # persona strategy and the RL mode vector read mtf_bias /
    # mtf_confluence_score / htf_* columns straight off these bars.
    dataset = _inject_mtf_features(dataset, symbol, timeframe, root)

    manifest = save_aligned_dataset(
        dataset,
        output,
        source_bars_file=bars_file,
        source_ticks_file=source_ticks_file,
        copy_ticks=False,
    )
    render_manifest(manifest, console=console)
    return TaskResult(True, f"aligned dataset ready at {output}", {"bars_file": manifest.files["bars"], "rows": len(dataset.bars)})


def _inject_mtf_features(dataset, symbol: str, timeframe: str, root: str | Path):
    """Merge higher-timeframe ICT features into the aligned M1 bars.

    Loads the collected higher-timeframe bars (M5, M15, H1, H4, D1) from MT5
    storage and computes per-bar MTF context (htf_* features, mtf_bias,
    mtf_confluence_score) causally aligned to the execution timeframe. If no
    higher-timeframe bars are available, the dataset is returned unchanged.
    """
    from dataclasses import replace

    from slytrade.config.mtf import get_higher_timeframes
    from slytrade.features.mtf import compute_mtf_ict_features

    higher: dict[str, pd.DataFrame] = {}
    for tf in get_higher_timeframes(timeframe):
        if tf == timeframe:
            continue
        try:
            higher[tf] = load_collected_bars(symbol, tf, root)
        except FileNotFoundError:
            continue
    if not higher:
        console.print("[yellow]No higher-timeframe bars found; skipping MTF feature injection.[/yellow]")
        return dataset

    try:
        mtf_bars = compute_mtf_ict_features(dataset.bars, higher)
    except Exception as exc:  # pragma: no cover - data dependent
        console.print(f"[yellow]MTF feature injection failed ({exc}); keeping single-timeframe bars.[/yellow]")
        return dataset
    console.print(f"[green]Injected MTF features from {', '.join(sorted(higher))} ({len(mtf_bars)} bars)[/green]")
    return replace(dataset, bars=mtf_bars)


def _load_frame(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _load_bars_or_error(bars_file: str | Path) -> tuple[pd.DataFrame | None, TaskResult | None]:
    """Load a bars file, returning a friendly TaskResult when it is missing.

    Prevents a raw FileNotFoundError traceback when the user runs a training/
    backtest command before the collection+alignment step has produced the
    aligned bars file.
    """
    from slytrade.backtest.reporting import load_bars_file

    path = Path(bars_file)
    if not path.exists():
        return None, TaskResult(
            False,
            f"bars file not found: {path}.\n"
            "Run the pipeline first to create it:\n"
            "  slytrade full-pipeline --symbol XAUUSD --source hybrid --lookback 1y",
        )
    try:
        return load_bars_file(path), None
    except Exception as exc:
        return None, TaskResult(False, f"failed to load bars file {path}: {exc}")


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
    commission: float | None = None,
    slippage_points: float | None = None,
) -> TaskResult:
    from slytrade.backtest.engine import BacktestConfig
    from slytrade.backtest.reporting import (
        infer_symbol,
        render_backtest_report,
        run_managed_aligned_backtest_from_bars,
    )
    from slytrade.progress import stage

    stage("Backtest (persona-adaptive, managed exits)")
    bars, load_error = _load_bars_or_error(bars_file)
    if load_error is not None:
        return load_error
    assert bars is not None
    resolved = infer_symbol(bars, symbol)
    point_value = point_value if point_value is not None else default_point_value(resolved)

    # Cost-aware defaults from configs/risk.yaml so the reported edge is NET of
    # commission + slippage (GAP-1). Explicit CLI values always win.
    if commission is None or slippage_points is None:
        from slytrade.core.config import load_config as _load_config

        risk_cfg = _load_config("configs").risk
        costs = risk_cfg.get("costs", {})
        if commission is None:
            commission = float(costs.get("commission_per_volume", 0.0) or 0.0)
        if slippage_points is None:
            slippage_points = float(costs.get("slippage_points", 0.0) or 0.0)

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
            slippage_points=slippage_points,
        ),
    )
    render_backtest_report(result, strategy_name=strategy, console=console)
    return TaskResult(
        True,
        "backtest complete (net of costs)",
        {
            "total_return": result.metrics.total_return,
            "max_drawdown": result.metrics.max_drawdown,
            "sharpe_like": result.metrics.sharpe_like,
            "trades": result.metrics.trades,
            "final_equity": result.metrics.final_equity,
            "commission_per_volume": commission,
            "slippage_points": slippage_points,
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
    reward: str = "r_multiple",
    artifacts_dir: str | Path = "models/artifacts",
    registry_path: str | Path = "models/registry.jsonl",
    n_envs: int = 1,
) -> TaskResult:
    from slytrade.backtest.reporting import infer_symbol
    from slytrade.progress import info, stage
    from slytrade.rl.dataset import build_rl_dataset
    from slytrade.rl.deployment import save_model_artifact
    from slytrade.rl.tracking import maybe_end_run, maybe_log_metrics, maybe_log_params, maybe_start_run
    from slytrade.rl.walkforward import evaluate_policy, resolve_algorithm, train_policy, train_ppo

    stage("Train RL policy (adopts the validated feature stack)")
    # The model artifact and registry live under models/ — make sure that tree
    # exists and is writable before a long training run starts.
    dir_error = _ensure_writable_root(Path(artifacts_dir).parent) or _ensure_writable_root(Path(registry_path).parent)
    if dir_error is not None:
        return dir_error

    algorithm = resolve_algorithm(algorithm)
    bars, load_error = _load_bars_or_error(bars_file)
    if load_error is not None:
        return load_error
    assert bars is not None
    resolved = infer_symbol(bars, symbol)
    bars = bars[bars["symbol"] == resolved].copy()
    try:
        dataset = build_rl_dataset(bars)
        info(f"dataset: {len(dataset.bars):,} bars × {len(dataset.features.columns)} features "
             f"(ML + ICT + tick microstructure + MTF)")
        scaler_params = dataset.fit_scaler(0, len(dataset.bars))
        # Dynamic footprint-driven feature selection (train slice only).
        selected = list(dataset.select_features_on_fold(0, len(dataset.bars)))
        info(f"dynamic selection: {len(selected)}/{len(dataset.features.columns)} features "
             f"(footprint significance, threshold-free)")
        env = dataset.env_factory(0, len(dataset.bars), seed=seed, scaler_params=scaler_params, feature_columns=selected)
    except ImportError as exc:
        return TaskResult(False, f"RL dependencies not installed: {exc}")
    env.config = _with_reward(env.config, reward)
    info(f"reward: {reward} | managed exits: {env.config.use_managed_exits} "
         f"(SL {env.config.stop_loss_atr}×ATR, TP {env.config.take_profit_atr}×ATR) | "
         f"episode length: {env.config.episode_length_bars} bars")

    # Model ids are unique per training run: the registry is append-only and
    # refuses duplicates, so a deterministic id would collide on every re-run
    # (exactly what broke the second pipeline run).
    from datetime import UTC, datetime

    # Microsecond resolution makes collisions between successive runs
    # effectively impossible.
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    model_id = f"{algorithm}-{resolved}-{seed}-{run_stamp}"

    run = maybe_start_run("slytrade-rl", run_name=model_id)
    maybe_log_params(run, {"algorithm": algorithm, "symbol": resolved, "seed": seed, "timesteps": total_timesteps, "policy": policy, "reward": reward, "features": len(dataset.features.columns)})
    try:
        if algorithm == "ppo":
            model = train_ppo(
                env,
                total_timesteps=total_timesteps,
                seed=seed,
                policy_type=policy,
                progress_bar=True,
                n_envs=n_envs,
            )
        else:
            model = train_policy(algorithm, env, total_timesteps=total_timesteps, seed=seed, policy_type=policy, progress_bar=True)
        results = evaluate_policy(model, env, episodes=3, seed=seed)
        maybe_log_metrics(run, {f"mean_{key}": float(value) for key, value in results.items() if isinstance(value, (int, float))})
    finally:
        maybe_end_run(run)

    record = save_model_artifact(
        model,
        model_id=model_id,
        algorithm=algorithm,
        symbol=resolved,
        feature_columns=selected,
        scaler_params=scaler_params,
        env_config={
            "reward_type": reward,
            "policy": policy,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "use_managed_exits": env.config.use_managed_exits,
            "n_features": len(dataset.features.columns),
        },
        metrics={f"mean_{key}": float(value) for key, value in results.items() if isinstance(value, (int, float))},
        artifacts_dir=artifacts_dir,
        registry_path=registry_path,
    )
    console.print(f"[green]Trained {algorithm.upper()} ({policy}) policy; artifact registered as {record['model_id']}[/green]")
    for key, value in results.items():
        console.print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    return TaskResult(True, "training complete", {"model_id": record["model_id"], "metrics": {k: float(v) for k, v in results.items() if isinstance(v, (int, float))}})


VALID_REWARDS = ("raw", "risk_adjusted", "trade_pnl", "r_multiple")


def _with_reward(config, reward: str):
    normalized = str(reward).strip().lower()
    if normalized not in VALID_REWARDS:
        raise ValueError(f"unknown reward type {reward!r}; choose from {VALID_REWARDS}")
    from dataclasses import replace

    return replace(config, reward_type=normalized)


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
    reward: str = "r_multiple",
    policy: str = "mlp",
    n_envs: int = 1,
) -> TaskResult:
    from slytrade.backtest.reporting import infer_symbol
    from slytrade.progress import info, stage
    from slytrade.rl.dataset import build_rl_dataset
    from slytrade.rl.walkforward import make_walk_forward_folds, resolve_fold_windows, walk_forward_validation

    stage("Walk-forward validation (embargoed out-of-sample)")
    bars, load_error = _load_bars_or_error(bars_file)
    if load_error is not None:
        return load_error
    assert bars is not None
    resolved = infer_symbol(bars, symbol)
    bars = bars[bars["symbol"] == resolved].copy()
    try:
        dataset = build_rl_dataset(bars)
        windows = resolve_fold_windows(
            len(dataset.bars),
            train_window=train_window,
            validation_window=validation_window,
            test_window=test_window,
            embargo=embargo,
        )
        if windows.train_window != train_window:
            console.print(
                f"[yellow]Dataset has {len(dataset.bars)} bars; scaling walk-forward windows to "
                f"train={windows.train_window} val={windows.validation_window} test={windows.test_window}.[/yellow]"
            )
        folds = make_walk_forward_folds(
            len(dataset.bars),
            train_window=windows.train_window,
            validation_window=windows.validation_window,
            test_window=windows.test_window,
            embargo=windows.embargo,
            step=windows.step,
        )
        info(f"{len(folds)} folds · reward={reward} · policy={policy} · {total_timesteps} steps/fold")
        table = walk_forward_validation(
            dataset, folds, total_timesteps=total_timesteps, seed=seed, reward_type=reward, policy_type=policy,
            progress=True, progress_bar=True, n_envs=n_envs,
        )
        if len(folds) < 2:
            console.print(
                "[yellow]Only one fold — this is a statistically weak read. Re-run with smaller windows "
                "(e.g. --train-window 60000 --validation-window 15000 --test-window 15000) to get multiple folds.[/yellow]"
            )
        table = _add_champion_comparison(table, dataset, folds, resolved)
    except ImportError as exc:
        return TaskResult(False, f"RL dependencies not installed: {exc}")
    console.print(table.to_string(index=False))
    return TaskResult(True, "walk-forward validation complete", {"folds": len(folds)})


def _add_champion_comparison(table: pd.DataFrame, dataset, folds, symbol: str) -> pd.DataFrame:
    """Append the persona-adaptive (rule-based champion) return per fold.

    The RL only earns a place in deployment if it beats the champion on the
    SAME out-of-sample windows. This puts that comparison in the report.
    """
    from slytrade.backtest.engine import BacktestConfig
    from slytrade.backtest.reporting import run_managed_aligned_backtest_from_bars
    from slytrade.backtest.trade_management import TradeManagementConfig

    persona_returns: list[float | None] = []
    for fold in folds:
        test_bars = dataset.bars.iloc[fold.test_start:fold.test_end].reset_index(drop=True)
        try:
            result = run_managed_aligned_backtest_from_bars(
                test_bars,
                strategy_name="persona-adaptive",
                symbol=symbol,
                volume=0.1,
                point_value=default_point_value(symbol),
                config=BacktestConfig(
                    initial_balance=100_000.0,
                    point_size=0.01,
                    point_value=default_point_value(symbol),
                    commission_per_volume=0.0,
                ),
                trade_config=TradeManagementConfig(stop_loss_atr=1.0, take_profit_atr=2.0),
            )
            realized = sum(float(record.realized_pnl) for record in result.trades if record.reason.startswith("managed_"))
            persona_returns.append(realized / 100_000.0)
        except Exception:  # pragma: no cover - data dependent
            persona_returns.append(None)

    valid = [value for value in persona_returns if value is not None]
    aggregate_persona = sum(valid) / len(valid) if valid else float("nan")

    rl_returns = table["test_mean_total_return"].tolist()
    n_folds = len(folds)
    rl_fold = rl_returns[:n_folds]
    rl_minus = [
        rl_fold[i] - (persona_returns[i] if persona_returns[i] is not None else float("nan"))
        for i in range(n_folds)
    ]
    valid_diff = [value for value in rl_minus if value == value]
    aggregate_diff = sum(valid_diff) / len(valid_diff) if valid_diff else float("nan")

    table = table.copy()
    table["persona_return"] = persona_returns + [aggregate_persona]
    table["rl_minus_persona"] = rl_minus + [aggregate_diff]
    return table


# ---------------------------------------------------------------------------
# Task: promote
# ---------------------------------------------------------------------------


def promote(model_id: str, *, stage: str = "paper", registry_path: str | Path = "models/registry.jsonl") -> TaskResult:
    from slytrade.rl.deployment import promote_artifact

    dir_error = _ensure_writable_root(Path(registry_path).parent)
    if dir_error is not None:
        return dir_error

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
    reward: str = "r_multiple",
    promote_stage: str = "paper",
    clean: bool = False,
    n_envs: int = 1,
) -> TaskResult:
    """Run collection → alignment → backtest → train → walk-forward → promote.

    With ``clean=True`` the derived data tree is wiped first so no stale files
    from a previous run can leak into this one.
    """
    from slytrade.progress import stage

    steps: list[str] = []
    stage(f"Full pipeline — {symbol} · lookback={lookback} · source={source}")
    if clean:
        cleaned = clean_all()
        if not cleaned.ok:
            return cleaned
        steps.append("clean")
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

    trained = train(bars_file, symbol=symbol, algorithm=algorithm, total_timesteps=total_timesteps, policy=policy, reward=reward, n_envs=n_envs)
    if not trained.ok:
        return trained
    steps.append("train")

    walk_forward(bars_file, symbol=symbol, reward=reward, policy=policy, n_envs=n_envs)
    steps.append("walk_forward")

    model_id = trained.data["model_id"] if trained.data else None
    if model_id:
        promote(model_id, stage=promote_stage)
        steps.append("promote")

    console.print(f"[bold green]Full pipeline complete: {' → '.join(steps)}[/bold green]")
    return TaskResult(True, "full pipeline complete", {"steps": steps, "model_id": model_id})


# ---------------------------------------------------------------------------
# Task: clean (reset derived data without touching directory ownership)
# ---------------------------------------------------------------------------

# Derived directories whose CONTENTS are wiped by clean_all(). The directories
# themselves are kept (and recreated) so their owner stays the current user —
# this is what makes repeated pipeline runs safe without sudo.
CLEANABLE_DIRS = (
    "data/raw",
    "data/processed",
    "data/exness_derived",
    "data/samples",
    "models",
    "logs",
    "state",
)


def clean_all() -> TaskResult:
    """Delete derived data from previous runs, keeping the directories.

    Removes the *contents* of the data/output directories so a fresh pipeline
    run never mixes stale files with new ones, but keeps the directories
    themselves owned by the current user (the root-cause fix for the
    "data is owned by root" permission loop caused by `sudo rm -rf data`).
    """
    import shutil

    removed = 0
    errors: list[str] = []
    for directory in CLEANABLE_DIRS:
        path = Path(directory)
        if not path.exists():
            continue
        try:
            for child in path.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed += 1
        except PermissionError:
            errors.append(f"'{path}' is owned by another user (e.g. root); run: sudo rm -rf \"{path}\"")
        except OSError as exc:  # pragma: no cover
            errors.append(f"'{path}': {exc}")
    if errors:
        return TaskResult(False, "some directories could not be cleaned:\n" + "\n".join(errors))

    # Recreate the standard tree (owned by the current user).
    dir_error = _ensure_standard_dirs()
    if dir_error is not None:
        return dir_error
    console.print(f"[green]Cleaned {removed} items; data tree is fresh.[/green]")
    return TaskResult(True, "clean complete", {"removed": removed})


# ---------------------------------------------------------------------------
# Task: robustness evidence (Monte Carlo + perturbation + regime segmentation)
# ---------------------------------------------------------------------------


def robustness(
    bars_file: str,
    *,
    strategy: str = "persona-adaptive",
    symbol: str | None = None,
    n_simulations: int = 2000,
) -> TaskResult:
    """Produce robustness evidence for a strategy on an aligned dataset.

    * Trade-sequence Monte Carlo of the realized-PnL sequence.
    * Parameter perturbation of the key risk knobs (SL/TP ATR multiples).
    * Regime segmentation of realized PnL (volatility/trend/session).
    """
    from slytrade.backtest.reporting import infer_symbol, load_bars_file
    from slytrade.progress import info, stage
    from slytrade.rl.robustness import (
        RobustnessReport,
        monte_carlo_trades,
        perturbation_sweep,
        regime_segmentation,
    )

    stage("Robustness evidence (Monte Carlo + perturbation + regime)")
    bars = load_bars_file(Path(bars_file))
    resolved = infer_symbol(bars, symbol)
    bars = bars[bars["symbol"] == resolved].copy()

    # Baseline backtest + its realized-PnL sequence.
    result = _run_backtest_for_pnls(bars, strategy, resolved)
    pnls = [float(record.realized_pnl) for record in result.trades if record.reason.startswith("managed_")]

    if not pnls:
        return TaskResult(False, "no closed trades to analyse; the strategy found no entries in this window")

    mc = monte_carlo_trades(pnls, n_simulations=n_simulations)
    info(f"observed total PnL {mc.observed_total:,.2f} across {len(pnls)} trades")
    info(f"Monte Carlo ({mc.n_simulations} resamples): mean {mc.mean_total:,.2f}, "
         f"95% CI [{mc.ci_95_low:,.2f}, {mc.ci_95_high:,.2f}], P(loss) {mc.prob_loss:.1%}")

    # Perturbation: re-run the same backtest with SL/TP ATR multiples shifted.
    base = {"stop_loss_atr": 1.0, "take_profit_atr": 2.0}

    def score(params):
        score_result = _run_backtest_for_pnls(bars, strategy, resolved, **params)
        return float(sum(float(record.realized_pnl) for record in score_result.trades if record.reason.startswith("managed_")))

    perturbations = perturbation_sweep(
        score,
        base,
        deltas={"stop_loss_atr": (-0.5, 0.0, 0.5), "take_profit_atr": (-0.5, 0.0, 0.5)},
    )
    for perturbation in perturbations:
        flag = "[yellow]SENSITIVE[/yellow]" if perturbation.sensitive else "[green]stable[/green]"
        info(f"perturb {perturbation.param}: scores {[round(s, 1) for s in perturbation.scores]} → {flag}")

    segments = regime_segmentation(pd.DataFrame(result.trades), bars)
    for segment in segments:
        info(f"regime '{segment.label}': {segment.trades} trades, total {segment.total_pnl:,.2f}, "
             f"win {segment.win_rate:.0%}")

    report = RobustnessReport(monte_carlo=mc, perturbations=perturbations, regimes=segments)
    console.print("[green]Robustness report ready[/green]")
    return TaskResult(True, "robustness complete", report.as_dict())


def _run_backtest_for_pnls(
    bars: pd.DataFrame,
    strategy: str,
    symbol: str,
    *,
    stop_loss_atr: float = 1.0,
    take_profit_atr: float = 2.0,
):
    """Run a managed backtest and return the BacktestResult (for robustness)."""
    from slytrade.backtest.engine import BacktestConfig
    from slytrade.backtest.reporting import run_managed_aligned_backtest_from_bars
    from slytrade.backtest.trade_management import TradeManagementConfig

    return run_managed_aligned_backtest_from_bars(
        bars,
        strategy_name=strategy,
        symbol=symbol,
        volume=0.1,
        point_value=default_point_value(symbol),
        config=BacktestConfig(
            initial_balance=100_000.0,
            point_size=0.01,
            point_value=default_point_value(symbol),
            commission_per_volume=0.0,
        ),
        trade_config=TradeManagementConfig(stop_loss_atr=stop_loss_atr, take_profit_atr=take_profit_atr),
    )
