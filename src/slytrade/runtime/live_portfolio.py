"""Multi-symbol LIVE portfolio.

Runs one guarded :class:`LiveTradingLoop` per symbol in parallel threads, all
against the SAME MT5 account, sharing a single :class:`PortfolioBreaker` so a
correlated drawdown across symbols can never exceed the portfolio risk budget.

Each loop keeps its own guardrails/breaker/strategy (per-symbol), journals into
the shared trades.parquet (thread-safe), and publishes a per-symbol status file
``state/live_status_<symbol>.json``. A background aggregator merges those into
the single ``state/live_status.json`` the dashboard reads.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slytrade.runtime.demo_loop import LiveTradingLoop
from slytrade.runtime.portfolio_loop import PortfolioBreaker
from slytrade.runtime.settings import RuntimeSettings


@dataclass
class LivePortfolio:
    """Spawn and supervise one LiveTradingLoop per symbol (live, real orders)."""

    symbols: list[str]
    settings: RuntimeSettings
    _threads: dict[str, threading.Thread] = field(default_factory=dict)
    _loops: dict[str, LiveTradingLoop] = field(default_factory=dict)
    _stop: threading.Event = field(default_factory=threading.Event)
    _breaker: PortfolioBreaker | None = field(default=None, init=False)

    def start(self, mt5) -> None:
        from slytrade.core.config import load_config

        risk_cfg = load_config(self.settings.config_dir).risk
        # The account equity is shared across symbols, so the portfolio's risk
        # budget is the whole account (initial_balance is the fallback).
        portfolio_balance = max(float(self.settings.initial_balance), 1.0) * max(len(self.symbols), 1)
        self._breaker = PortfolioBreaker(
            portfolio_balance,
            max_daily_drawdown=float(risk_cfg.get("max_daily_drawdown", 0.03) or 0.03),
            max_total_drawdown=float(risk_cfg.get("max_total_drawdown", 0.08) or 0.08),
        )
        for symbol in self.symbols:
            settings = RuntimeSettings(**{**self.settings.model_dump(), "symbol": symbol})
            loop = LiveTradingLoop(settings, mt5, portfolio_breaker=self._breaker, status_key=symbol)
            self._loops[symbol] = loop
            thread = threading.Thread(target=loop.run, name=f"live-{symbol}", daemon=True)
            self._threads[symbol] = thread
            thread.start()
        threading.Thread(target=self._aggregate_loop, name="live-portfolio-aggregator", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        for loop in self._loops.values():
            loop.stop()
        for thread in self._threads.values():
            thread.join(timeout=5.0)

    # -- status aggregation --------------------------------------------------
    def _aggregate_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._publish_aggregate()
            except Exception:  # pragma: no cover - aggregation must never crash the book
                pass
            time.sleep(3.0)

    def _publish_aggregate(self) -> None:
        state_dir = Path(self.settings.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        per_symbol: dict[str, dict[str, Any]] = {}
        for symbol in self.symbols:
            path = state_dir / f"live_status_{symbol}.json"
            if not path.exists():
                continue
            try:
                per_symbol[symbol] = json.loads(path.read_text())
            except Exception:  # pragma: no cover
                continue
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "mode": "portfolio",
            "symbols": self.symbols,
            "count": len(self.symbols),
            "portfolio_breaker_tripped": bool(self._breaker.tripped) if self._breaker else False,
            "portfolio_breaker_reason": self._breaker.reason if self._breaker else None,
            "per_symbol": per_symbol,
        }
        # Backward-compatible top-level fields (the dashboard reads these): show
        # the first symbol's live snapshot plus the joined watchlist.
        if per_symbol:
            first = next(iter(per_symbol.values()))
            for key in (
                "timeframe", "price", "side", "pending_limit", "equity", "balance",
                "errors", "orders", "fills", "kill_switch", "last_decision", "bars_built", "tick",
            ):
                if key in first:
                    payload[key] = first[key]
            payload["symbol"] = ", ".join(self.symbols)
        tmp = state_dir / "live_status.json.tmp"
        tmp.write_text(json.dumps(payload, default=str))
        os.replace(tmp, state_dir / "live_status.json")

    def summaries(self) -> dict[str, object]:
        return {
            symbol: {"breaker_tripped": bool(self._breaker.tripped) if self._breaker else False}
            for symbol in self.symbols
        }
