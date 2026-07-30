from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import pandas as pd
from rich.console import Console
from rich.table import Table

from slytrade.backtest.engine import BacktestConfig, BacktestResult, BarBacktestEngine, BarStrategy
from slytrade.features.ict import FEATURE_COLUMNS, compute_ict_features
from slytrade.strategies.baselines import (
    BuyAndHoldStrategy,
    ICTBiasBaselineStrategy,
    MovingAverageCrossStrategy,
    NoTradeStrategy,
)

StrategyName = Literal["no-trade", "buy-and-hold", "ma-cross", "ict-bias"]
VALID_STRATEGIES: tuple[str, ...] = ("no-trade", "buy-and-hold", "ma-cross", "ict-bias")


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


def metrics_as_dict(result: BacktestResult) -> dict[str, float | int]:
    return asdict(result.metrics)


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
