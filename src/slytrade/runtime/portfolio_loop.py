"""Multi-symbol paper portfolio.

Runs one supervised paper-trading loop per symbol in parallel threads, sharing a
single Prometheus registry and a single metric server. Each symbol keeps its own
guardrails, breaker and ledger, so a drawdown breach on one symbol never stops
the others.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

from slytrade.runtime.metrics_server import TradingMetrics
from slytrade.runtime.paper_loop import MT5QuoteProvider, PaperTradingLoop
from slytrade.runtime.settings import RuntimeSettings


class PortfolioBreaker:
    """Portfolio-level circuit breaker shared across all symbol loops.

    Aggregates realised PnL across symbols (thread-safe) and trips a shared
    kill when the TOTAL portfolio breaches the same daily/total drawdown limits
    the per-symbol breakers use. This is the guard against correlated losses —
    the classic multi-symbol failure mode (every symbol short USD, every symbol
    long gold, and the whole book blows up together).
    """

    def __init__(
        self,
        portfolio_balance: float,
        *,
        max_daily_drawdown: float = 0.03,
        max_total_drawdown: float = 0.08,
    ) -> None:
        self._lock = threading.Lock()
        self._balance = max(float(portfolio_balance), 1e-9)
        self._realized = 0.0
        self._peak = 0.0
        self._daily = 0.0
        self._day: str | None = None
        self.max_daily_drawdown = float(max_daily_drawdown)
        self.max_total_drawdown = float(max_total_drawdown)
        self.tripped = False
        self.reason: str | None = None

    def record(self, symbol: str, realized: float) -> None:
        with self._lock:
            self._realized += float(realized)
            self._peak = max(self._peak, self._realized)
            today = datetime.now(UTC).date().isoformat()
            if self._day != today:
                self._day = today
                self._daily = 0.0
            self._daily += float(realized)

    def allowed(self) -> bool:
        with self._lock:
            total_dd = (self._peak - self._realized) / self._balance
            if total_dd >= self.max_total_drawdown:
                self.tripped = True
                self.reason = f"portfolio total drawdown {total_dd:.1%} >= {self.max_total_drawdown:.1%}"
                return False
            daily_dd = -self._daily / self._balance
            if daily_dd >= self.max_daily_drawdown:
                self.tripped = True
                self.reason = f"portfolio daily drawdown {daily_dd:.1%} >= {self.max_daily_drawdown:.1%}"
                return False
            return True


@dataclass
class PaperPortfolio:
    """Spawn and supervise one PaperTradingLoop per symbol.

    Sizing and kill-switches stay per-symbol; a shared :class:`PortfolioBreaker`
    additionally halts the WHOLE book on an aggregate drawdown, so a correlated
    multi-symbol loss can never exceed the portfolio risk budget.
    """

    symbols: list[str]
    settings: RuntimeSettings
    metrics: TradingMetrics = field(default_factory=TradingMetrics)
    _threads: dict[str, threading.Thread] = field(default_factory=dict)
    _loops: dict[str, PaperTradingLoop] = field(default_factory=dict)

    def start(self, mt5) -> None:
        portfolio_balance = max(float(self.settings.initial_balance), 1.0) * max(len(self.symbols), 1)
        from slytrade.core.config import load_config

        risk_cfg = load_config(self.settings.config_dir).risk
        breaker = PortfolioBreaker(
            portfolio_balance,
            max_daily_drawdown=float(risk_cfg.get("max_daily_drawdown", 0.03) or 0.03),
            max_total_drawdown=float(risk_cfg.get("max_total_drawdown", 0.08) or 0.08),
        )
        for symbol in self.symbols:
            settings = RuntimeSettings(**{**self.settings.model_dump(), "symbol": symbol})
            provider = MT5QuoteProvider(symbol, mt5, poll_seconds=settings.poll_seconds)
            loop = PaperTradingLoop(settings, provider, portfolio_breaker=breaker)
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
