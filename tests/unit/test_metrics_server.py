from __future__ import annotations

from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from slytrade.runtime.metrics_server import MetricsServer, TradingMetrics


@pytest.fixture()
def server():
    metrics = TradingMetrics()
    metrics.equity.set(123.0)
    ready = {"flag": True}
    srv = MetricsServer(port=0, bind="127.0.0.1", metrics=metrics, readiness=lambda: (ready["flag"], "not ready"))
    srv.start()
    yield srv, metrics, ready
    srv.stop()


def test_metrics_endpoint(server) -> None:
    srv, _, _ = server
    with urlopen(f"http://127.0.0.1:{srv.bound_port}/metrics", timeout=5) as response:
        body = response.read().decode()
    assert response.status == 200
    assert "slytrade_equity" in body
    assert "slytrade_orders_total" in body


def test_healthz_ok(server) -> None:
    srv, _, _ = server
    with urlopen(f"http://127.0.0.1:{srv.bound_port}/healthz", timeout=5) as response:
        assert response.status == 200
        assert response.read() == b"ok\n"


def test_readyz_reflects_readiness(server) -> None:
    srv, _, ready = server
    with urlopen(f"http://127.0.0.1:{srv.bound_port}/readyz", timeout=5) as response:
        assert response.status == 200
    ready["flag"] = False
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"http://127.0.0.1:{srv.bound_port}/readyz", timeout=5)
    assert exc_info.value.code == 503


def test_unknown_path_404(server) -> None:
    srv, _, _ = server
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"http://127.0.0.1:{srv.bound_port}/nope", timeout=5)
    assert exc_info.value.code == 404
