"""Prometheus metrics + Kubernetes-style liveness/readiness endpoints.

A single lightweight HTTP server exposes:

* ``GET /metrics`` — Prometheus text exposition (scraped by Prometheus).
* ``GET /healthz`` — liveness: the process is up and the server is serving.
* ``GET /readyz``  — readiness: the trading loop has observed healthy data and
  the risk state is not in a hard-stop condition.

The server is intentionally tiny and dependency-light (stdlib HTTP server +
``prometheus_client``) so it can run as the container's main health surface or
as a sidecar.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


@dataclass
class TradingMetrics:
    """One place for every runtime metric the loop updates."""

    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        self.orders_total = Counter(
            "slytrade_orders_total",
            "Orders submitted to the paper broker, by status.",
            ["status"],
            registry=self.registry,
        )
        self.trades_total = Counter(
            "slytrade_trades_total",
            "Closed trades, by outcome (win/loss/breakeven).",
            ["outcome"],
            registry=self.registry,
        )
        self.broker_errors_total = Counter(
            "slytrade_broker_errors_total",
            "Broker/quote provider errors.",
            registry=self.registry,
        )
        self.stale_quotes_total = Counter(
            "slytrade_stale_quotes_total",
            "Quotes skipped because they were stale.",
            registry=self.registry,
        )
        self.execution_latency_seconds = Histogram(
            "slytrade_execution_latency_seconds",
            "End-to-end decision latency per bar.",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )
        self.equity = Gauge("slytrade_equity", "Current paper equity.", registry=self.registry)
        self.balance = Gauge("slytrade_balance", "Current paper balance.", registry=self.registry)
        self.open_positions = Gauge(
            "slytrade_open_positions",
            "Number of currently open positions.",
            registry=self.registry,
        )
        self.daily_drawdown = Gauge(
            "slytrade_daily_drawdown",
            "Current daily drawdown (0..1).",
            registry=self.registry,
        )
        self.total_drawdown = Gauge(
            "slytrade_total_drawdown",
            "Current drawdown from peak equity (0..1).",
            registry=self.registry,
        )
        self.kill_switch = Gauge(
            "slytrade_kill_switch",
            "1 when the kill switch is active, else 0.",
            registry=self.registry,
        )
        self.trading_paused = Gauge(
            "slytrade_trading_paused",
            "1 when the circuit breaker has paused new entries.",
            registry=self.registry,
        )
        self.news_paused = Gauge(
            "slytrade_news_paused",
            "1 when the red-folder news gate is pausing new entries.",
            registry=self.registry,
        )
        self.news_pauses_total = Counter(
            "slytrade_news_pauses_total",
            "Bars where new entries were paused by the red-folder news gate.",
            registry=self.registry,
        )
        self.uptime_seconds = Gauge("slytrade_uptime_seconds", "Process uptime.", registry=self.registry)


class MetricsServer:
    """Threaded HTTP server serving /metrics, /healthz and /readyz."""

    def __init__(
        self,
        *,
        port: int,
        bind: str = "0.0.0.0",
        metrics: TradingMetrics | None = None,
        readiness: Callable[[], tuple[bool, str]] | None = None,
        liveness: Callable[[], bool] | None = None,
    ) -> None:
        self.port = port
        self.bind = bind
        self.metrics = metrics or TradingMetrics()
        self.readiness = readiness or (lambda: (True, "ok"))
        self.liveness = liveness or (lambda: True)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        metrics = self.metrics
        readiness = self.readiness
        liveness = self.liveness

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:  # quiet default logging
                return

            def _write(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 (stdlib handler signature)
                if self.path == "/metrics":
                    body = generate_latest(metrics.registry)
                    self._write(200, body, CONTENT_TYPE_LATEST)
                elif self.path == "/healthz":
                    ok = liveness()
                    self._write(200 if ok else 503, b"ok\n" if ok else b"unhealthy\n", "text/plain")
                elif self.path == "/readyz":
                    ready, detail = readiness()
                    payload = ("ready\n" if ready else f"not ready: {detail}\n").encode()
                    self._write(200 if ready else 503, payload, "text/plain")
                else:
                    self._write(404, b"not found\n", "text/plain")

        self._server = ThreadingHTTPServer((self.bind, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="slytrade-metrics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def bound_port(self) -> int:
        """Actual port the server bound to (useful when started on port 0)."""
        if self._server is None:
            return self.port
        return int(self._server.server_address[1])
