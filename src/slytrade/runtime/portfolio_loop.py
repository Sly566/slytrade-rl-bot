"""Multi-symbol paper portfolio.

Runs one supervised paper-trading loop per symbol in parallel threads, sharing a
single Prometheus registry and a single metric server. Each symbol keeps its own
guardrails, breaker and ledger, so a drawdown breach on one symbol never stops
the others.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from slytrade.runtime.metrics_server import TradingMetrics
from slytrade.runtime.paper_loop import MT5QuoteProvider, PaperTradingLoop
from slytrade.runtime.settings import RuntimeSettings


@dataclass
class PaperPortfolio:
    """Spawn and supervise one PaperTradingLoop per symbol."""

    symbols: list[str]
    settings: RuntimeSettings
    metrics: TradingMetrics = field(default_factory=TradingMetrics)
    _threads: dict[str, threading.Thread] = field(default_factory=dict)
    _loops: dict[str, PaperTradingLoop] = field(default_factory=dict)

    def start(self, mt5) -> None:
        for symbol in self.symbols:
            settings = RuntimeSettings(**{**self.settings.model_dump(), "symbol": symbol})
            provider = MT5QuoteProvider(symbol, mt5, poll_seconds=settings.poll_seconds)
            loop = PaperTradingLoop(settings, provider)
            loop.metrics = self.metrics  # share one registry across symbols
            self._loops[symbol] = loop
            thread = threading.Thread(target=loop.run, name=f"paper-{symbol}", daemon=True)
            self._threads[symbol] = thread
            thread.start()

    def stop(self) -> None:
        for loop in self._loops.values():
            loop.stop()
        for thread in self._threads.values():
            thread.join(timeout=5.0)

    def summaries(self) -> dict[str, object]:
        return {
            symbol: loop._summary.__dict__ if loop._summary is not None else {"running": True}
            for symbol, loop in self._loops.items()
        }
