"""Dashboard + status-publishing tests: the platform the operator watches.

Covers the HTTP surface (health, auth, status, trades, logs, control) and the
live loop's atomic status-file contract that feeds it.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from slytrade.runtime.dashboard import DashboardServer, LoopSupervisor
from slytrade.runtime.demo_loop import LiveTradingLoop
from slytrade.runtime.settings import RuntimeSettings, TradingStage


class FakeMT5:
    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def symbols_get(self):
        return []

    def symbol_select(self, name: str, enable: bool = True) -> bool:
        return True

    def positions_get(self):
        return []

    def account_info(self):
        from types import SimpleNamespace

        return SimpleNamespace(equity=1000.0, balance=1000.0)


def _get(url: str, token: str | None = None) -> tuple[int, bytes]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post(url: str, payload: dict, token: str | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_dashboard_health_and_status(tmp_path) -> None:
    server = DashboardServer(host="127.0.0.1", port=0, state_dir=str(tmp_path), data_dir=str(tmp_path), log_dir=str(tmp_path))
    server.start()
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        code, _ = _get(base + "/healthz")
        assert code == 200
        code, _ = _get(base + "/readyz")
        assert code == 503  # no status file yet
        # write a status file, then readiness passes
        Path(tmp_path / "live_status.json").write_text(json.dumps({"ts": "x", "symbol": "XAUUSD"}))
        code, _ = _get(base + "/readyz")
        assert code == 200
        code, body = _get(base + "/api/status")
        assert code == 200
        payload = json.loads(body)
        assert payload["status"]["symbol"] == "XAUUSD"
        code, body = _get(base + "/")
        assert code == 200 and b"SlyTrade Control" in body
    finally:
        server.stop()


def test_dashboard_token_auth(tmp_path) -> None:
    server = DashboardServer(host="127.0.0.1", port=0, token="secret123", state_dir=str(tmp_path), data_dir=str(tmp_path), log_dir=str(tmp_path))
    server.start()
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        code, _ = _get(base + "/api/status")
        assert code == 401
        code, body = _get(base + "/api/status", token="wrong")
        assert code == 401
        code, body = _get(base + "/api/status", token="secret123")
        assert code == 200
        code, body = _get(base + "/api/status?token=secret123")
        assert code == 200
    finally:
        server.stop()


def test_dashboard_trades_reads_journal(tmp_path) -> None:
    journal_dir = tmp_path / "live_journal"
    journal_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"time": "2026-08-18T12:00:00", "side": "buy", "entry": 4000.0, "exit": 4009.0, "outcome_r": 3.0, "exit_reason": "take_profit", "volume": 0.1},
            {"time": "2026-08-18T14:00:00", "side": "sell", "entry": 4010.0, "exit": 4013.0, "outcome_r": -1.0, "exit_reason": "stop_loss", "volume": 0.1},
        ]
    ).to_parquet(journal_dir / "trades.parquet")
    server = DashboardServer(host="127.0.0.1", port=0, state_dir=str(tmp_path), data_dir=str(tmp_path), log_dir=str(tmp_path))
    server.start()
    try:
        code, body = _get(f"http://127.0.0.1:{server.bound_port}/api/trades")
        assert code == 200
        trades = json.loads(body)
        assert len(trades) == 2
        assert trades[0]["outcome_r"] == 3.0
        assert trades[1]["exit_reason"] == "stop_loss"
    finally:
        server.stop()


def test_supervisor_start_stop_and_logs(tmp_path) -> None:
    script = (
        "import time,sys; print('LOOP-UP'); sys.stdout.flush(); "
        "time.sleep(30)"
    )
    sup = LoopSupervisor([sys.executable, "-c", script], cwd=str(tmp_path), env=dict())
    sup.start()
    deadline = time.time() + 10
    while not sup.running and time.time() < deadline:
        time.sleep(0.05)
    assert sup.running
    deadline = time.time() + 10
    while "LOOP-UP" not in "\n".join(sup.tail(20)) and time.time() < deadline:
        time.sleep(0.05)
    assert "LOOP-UP" in "\n".join(sup.tail(20))
    sup.stop()
    deadline = time.time() + 10
    while sup.running and time.time() < deadline:
        time.sleep(0.05)
    assert not sup.running


def test_dashboard_control_endpoints(tmp_path) -> None:
    script = "import time; print('CHILD-READY'); time.sleep(30)"
    sup = LoopSupervisor([sys.executable, "-c", script], cwd=str(tmp_path), env=dict())
    server = DashboardServer(host="127.0.0.1", port=0, state_dir=str(tmp_path), data_dir=str(tmp_path), log_dir=str(tmp_path), supervisor=sup)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        code, body = _post(base + "/api/control", {"action": "start"})
        assert code == 200
        deadline = time.time() + 10
        while not sup.running and time.time() < deadline:
            time.sleep(0.05)
        assert sup.running
        code, body = _post(base + "/api/control", {"action": "stop"})
        assert code == 200
        deadline = time.time() + 10
        while sup.running and time.time() < deadline:
            time.sleep(0.05)
        assert not sup.running
        code, body = _post(base + "/api/control", {"action": "bogus"})
        assert code == 400
    finally:
        server.stop()


def test_live_loop_publishes_status_file(tmp_path) -> None:
    settings = RuntimeSettings(
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        kill_switch_path=str(tmp_path / "state" / "kill-switch.json"),
        json_logs=False,
        symbol="XAUUSD",
        timeframe="M15",
        allow_live=True,
        stage=TradingStage.DEMO,
    )
    loop = LiveTradingLoop(settings, FakeMT5())
    loop._ticks = 7
    loop._bar_index = 3
    loop._last_decision = "bar 3 → HOLD (below threshold)"
    loop._publish_status()
    path = Path(settings.state_dir) / "live_status.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["symbol"] == "XAUUSD"
    assert payload["tick"] == 7
    assert payload["bars_built"] == 3
    assert payload["last_decision"] == "bar 3 → HOLD (below threshold)"
    assert payload["journal_path"].endswith("trades.parquet")
