from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from rich.console import Console
from rich.table import Table

from slytrade.backtest.engine import BacktestConfig, BacktestResult, BarBacktestEngine, BarStrategy
from slytrade.backtest.tick_engine import TickBacktestEngine
from slytrade.features.ict import FEATURE_COLUMNS, compute_ict_features
from slytrade.strategies.baselines import (
    BuyAndHoldStrategy,
    ICTBiasBaselineStrategy,
    MovingAverageCrossStrategy,
    NoTradeStrategy,
)

StrategyName = Literal["no-trade", "buy-and-hold", "ma-cross", "ict-bias"]
VALID_STRATEGIES: tuple[str, ...] = ("no-trade", "buy-and-hold", "ma-cross", "ict-bias")


@dataclass(frozen=True)
class BaselineComparisonRow:
    strategy: str
    start_equity: float
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe_like: float
    trades: int
    orders: int
    ledger_records: int


def load_bars_file(path: str | Path) -> pd.DataFrame:
    """Load a canonical bar file from CSV or Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported bars file format: {path.suffix}. Use .csv or .parquet")


def load_ticks_file(path: str | Path) -> pd.DataFrame:
    """Load a canonical tick file from CSV or Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported ticks file format: {path.suffix}. Use .csv or .parquet")


def infer_symbol(bars: pd.DataFrame, symbol: str | None = None) -> str:
    if symbol:
        return symbol
    if "symbol" not in bars.columns or bars.empty:
        raise ValueError("symbol must be supplied when bars file is empty or lacks a symbol column")
    symbols = sorted(str(value) for value in bars["symbol"].dropna().unique())
    if len(symbols) != 1:
        raise ValueError(f"bars file must contain exactly one symbol or --symbol must be provided; found {symbols}")
    return symbols[0]


def ensure_ict_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Return bars with causal ICT feature columns available."""
    missing_features = [column for column in ["bos_dir", "choch_dir", "premium_discount", "liquidity_sweep"] if column not in bars.columns]
    if not missing_features:
        return bars.copy()
    features = compute_ict_features(bars)
    merged = bars.sort_values("time").reset_index(drop=True).copy()
    for column in FEATURE_COLUMNS:
        if column in features.columns:
            merged[column] = features[column].to_numpy()
    return merged


def build_strategy(
    strategy_name: str,
    *,
    symbol: str,
    volume: float,
    fast_window: int = 5,
    slow_window: int = 20,
) -> BarStrategy:
    if strategy_name == "no-trade":
        return NoTradeStrategy()
    if strategy_name == "buy-and-hold":
        return BuyAndHoldStrategy(symbol=symbol, volume=volume)
    if strategy_name == "ma-cross":
        return MovingAverageCrossStrategy(symbol=symbol, volume=volume, fast_window=fast_window, slow_window=slow_window)
    if strategy_name == "ict-bias":
        return ICTBiasBaselineStrategy(symbol=symbol, volume=volume)
    raise ValueError(f"Unknown strategy '{strategy_name}'. Valid strategies: {', '.join(VALID_STRATEGIES)}")


def run_backtest_from_bars(
    bars: pd.DataFrame,
    *,
    strategy_name: str,
    symbol: str | None = None,
    volume: float = 0.1,
    fast_window: int = 5,
    slow_window: int = 20,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    resolved_symbol = infer_symbol(bars, symbol)
    prepared_bars = ensure_ict_features(bars) if strategy_name == "ict-bias" else bars.copy()
    strategy = build_strategy(
        strategy_name,
        symbol=resolved_symbol,
        volume=volume,
        fast_window=fast_window,
        slow_window=slow_window,
    )
    engine = BarBacktestEngine(config)
    return engine.run(prepared_bars, strategy)


def run_tick_backtest_from_frames(
    bars: pd.DataFrame,
    ticks: pd.DataFrame,
    *,
    strategy_name: str,
    symbol: str | None = None,
    volume: float = 0.1,
    fast_window: int = 5,
    slow_window: int = 20,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    resolved_symbol = infer_symbol(bars, symbol)
    prepared_bars = ensure_ict_features(bars) if strategy_name == "ict-bias" else bars.copy()
    strategy = build_strategy(
        strategy_name,
        symbol=resolved_symbol,
        volume=volume,
        fast_window=fast_window,
        slow_window=slow_window,
    )
    engine = TickBacktestEngine(config)
    return engine.run(prepared_bars, ticks, strategy)


def metrics_as_dict(result: BacktestResult) -> dict[str, float | int]:
    return asdict(result.metrics)


def comparison_row(strategy_name: str, result: BacktestResult) -> BaselineComparisonRow:
    metrics = result.metrics
    return BaselineComparisonRow(
        strategy=strategy_name,
        start_equity=metrics.start_equity,
        final_equity=metrics.final_equity,
        total_return=metrics.total_return,
        max_drawdown=metrics.max_drawdown,
        sharpe_like=metrics.sharpe_like,
        trades=metrics.trades,
        orders=len(result.orders),
        ledger_records=len(result.trades),
    )


def compare_baselines_from_bars(
    bars: pd.DataFrame,
    *,
    symbol: str | None = None,
    volume: float = 0.1,
    fast_window: int = 5,
    slow_window: int = 20,
    config: BacktestConfig | None = None,
    strategies: tuple[str, ...] = VALID_STRATEGIES,
) -> list[BaselineComparisonRow]:
    rows: list[BaselineComparisonRow] = []
    for strategy_name in strategies:
        result = run_backtest_from_bars(
            bars,
            strategy_name=strategy_name,
            symbol=symbol,
            volume=volume,
            fast_window=fast_window,
            slow_window=slow_window,
            config=config,
        )
        rows.append(comparison_row(strategy_name, result))
    return sorted(rows, key=lambda row: row.final_equity, reverse=True)


def comparison_as_frame(rows: list[BaselineComparisonRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(row) for row in rows])


def render_backtest_report(
    result: BacktestResult,
    *,
    strategy_name: str,
    console: Console | None = None,
) -> None:
    target = console or Console()
    metrics = result.metrics
    table = Table(title=f"Backtest Report — {strategy_name}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Start Equity", f"{metrics.start_equity:,.2f}")
    table.add_row("Final Equity", f"{metrics.final_equity:,.2f}")
    table.add_row("Total Return", f"{metrics.total_return:.2%}")
    table.add_row("Max Drawdown", f"{metrics.max_drawdown:.2%}")
    table.add_row("Sharpe-like", f"{metrics.sharpe_like:.4f}")
    table.add_row("Equity Points", str(metrics.equity_points))
    table.add_row("Trades", str(metrics.trades))
    table.add_row("Orders", str(len(result.orders)))
    table.add_row("Ledger Records", str(len(result.trades)))
    target.print(table)


def render_baseline_comparison(rows: list[BaselineComparisonRow], *, console: Console | None = None) -> None:
    target = console or Console()
    table = Table(title="Baseline Comparison")
    table.add_column("Rank", justify="right")
    table.add_column("Strategy")
    table.add_column("Final Equity", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Sharpe-like", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Orders", justify="right")

    for rank, row in enumerate(rows, start=1):
        table.add_row(
            str(rank),
            row.strategy,
            f"{row.final_equity:,.2f}",
            f"{row.total_return:.2%}",
            f"{row.max_drawdown:.2%}",
            f"{row.sharpe_like:.4f}",
            str(row.trades),
            str(row.orders),
        )
    target.print(table)
