"""Layer 5 exports — streaming hedging backtest engine."""
from __future__ import annotations

from .engine import BacktestConfig, BacktestEngine, BacktestResult, run_backtest
from .positions import Direction, ExitReason, Position, Tranche, TrancheState
from .specs import AccountSpec, SymbolSpec, spec_for_symbol

__all__ = [
    "AccountSpec", "SymbolSpec", "spec_for_symbol",
    "Direction", "ExitReason", "Position", "Tranche", "TrancheState",
    "BacktestConfig", "BacktestEngine", "BacktestResult", "run_backtest",
]
