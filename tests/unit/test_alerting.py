from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from slytrade.runtime.alerting import (
    Alert,
    AlertManager,
    LogChannel,
    TelegramChannel,
    WebhookChannel,
)


class _CapturingServer:
    """Minimal local HTTP server that records requests."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                server.requests.append({"method": "POST", "path": self.path, "body": body})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def do_GET(self):  # noqa: N802
                server.requests.append({"method": "GET", "path": self.path, "body": b""})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *_: object) -> None:
                return

        return Handler

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def test_alert_payload() -> None:
    alert = Alert(severity="critical", title="kill switch", detail="daily drawdown breached")
    assert alert.severity == "critical"
    assert alert.created_at


def test_webhook_channel_posts_json() -> None:
    server = _CapturingServer()
    try:
        channel = WebhookChannel(f"http://127.0.0.1:{server.port}/hook")
        assert channel.send(Alert("critical", "kill switch", "drawdown")) is True
        assert len(server.requests) == 1
        payload = json.loads(server.requests[0]["body"])
        assert payload["severity"] == "critical"
        assert payload["title"] == "kill switch"
    finally:
        server.stop()


def test_webhook_channel_swallows_failures() -> None:
    channel = WebhookChannel("http://127.0.0.1:1/unreachable", timeout=1.0)
    # Must not raise even though nothing is listening.
    assert channel.send(Alert("info", "x")) is False


def test_webhook_rejects_bad_url() -> None:
    with pytest.raises(ValueError):
        WebhookChannel("ftp://nope")


def test_telegram_channel_builds_request() -> None:
    server = _CapturingServer()
    try:
        channel = TelegramChannel("token123", "chat456", base_url=f"http://127.0.0.1:{server.port}")
        assert channel.send(Alert("warning", "soak stale", "no heartbeat")) is True
        assert len(server.requests) == 1
        path = server.requests[0]["path"]
        assert "/bottoken123/sendMessage" in path
        assert "chat_id=chat456" in path
    finally:
        server.stop()


def test_telegram_channel_swallows_failures() -> None:
    channel = TelegramChannel("token", "chat", base_url="http://127.0.0.1:1", timeout=1.0)
    assert channel.send(Alert("info", "x")) is False


def test_alert_manager_never_raises() -> None:
    class Boom:
        def send(self, alert: Alert) -> bool:
            raise RuntimeError("channel exploded")

    manager = AlertManager([Boom(), LogChannel()])
    alert = manager.alert("critical", "should not crash")
    assert alert.title == "should not crash"


def test_alert_manager_default_channel_is_log() -> None:
    manager = AlertManager()
    assert manager.alert("info", "hello").title == "hello"
