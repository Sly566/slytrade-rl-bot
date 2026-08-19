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


def test_dashboard_multi_token_auth(tmp_path) -> None:
    """Comma-separated tokens = one capability per user; each works, none else."""
    server = DashboardServer(host="127.0.0.1", port=0, token="alice-tok,bob-tok", state_dir=str(tmp_path), data_dir=str(tmp_path), log_dir=str(tmp_path))
    server.start()
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        assert _get(base + "/api/status", token="alice-tok")[0] == 200
        assert _get(base + "/api/status", token="bob-tok")[0] == 200
        assert _get(base + "/api/status", token="eve-tok")[0] == 401
        assert _get(base + "/api/status")[0] == 401
    finally:
        server.stop()


def test_gen_token_command_prints_a_token(tmp_path) -> None:
    from typer.testing import CliRunner

    from slytrade.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["gen-token"])
    assert result.exit_code == 0
    assert "New dashboard token:" in result.output
    # token_urlsafe(32) is 43 chars of A-Za-z0-9_- — strip the label line and check length
    token_line = [line for line in result.output.splitlines() if "New dashboard token:" in line][0]
    token = token_line.split(":", 1)[1].strip()
    assert len(token) >= 32
    assert "," not in token  # no commas — safe to comma-separate lists


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


def _pipeline_server(tmp_path, token: str = ""):
    from slytrade.runtime.dashboard import PipelineRunner

    pipeline = PipelineRunner(cwd=str(tmp_path), env={})
    server = DashboardServer(
        host="127.0.0.1", port=0, token=token,
        state_dir=str(tmp_path), data_dir=str(tmp_path), log_dir=str(tmp_path),
        pipeline=pipeline,
    )
    return server, pipeline


def test_pipeline_overview_and_run(tmp_path) -> None:
    server, pipeline = _pipeline_server(tmp_path)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        # overview lists the six stages, none running
        code, body = _get(base + "/api/pipeline")
        assert code == 200
        tasks = json.loads(body)["tasks"]
        ids = [t["id"] for t in tasks]
        assert ids == ["full-pipeline", "collect", "admit", "backtest", "walk-forward", "learn"]
        assert all(not t["running"] for t in tasks)

        # run a fake task via direct runner (the real CLI needs data/MT5)
        ok, detail = pipeline.run("backtest", ["doctor"])
        assert ok
        deadline = time.time() + 10
        while not pipeline.running() and time.time() < deadline:
            time.sleep(0.05)
        assert pipeline.running() == ["backtest"]

        code, body = _get(base + "/api/pipeline")
        tasks = json.loads(body)["tasks"]
        backtest = [t for t in tasks if t["id"] == "backtest"][0]
        assert backtest["running"] is True

        # a second run is refused while busy
        ok2, detail2 = pipeline.run("learn", ["doctor"])
        assert not ok2 and "busy" in detail2

        # stop it
        code, body = _post(base + "/api/pipeline/stop", {"task": "backtest"})
        assert code == 200
        deadline = time.time() + 10
        while pipeline.running() and time.time() < deadline:
            time.sleep(0.05)
        assert not pipeline.running()
    finally:
        server.stop()


def test_pipeline_run_endpoint_validates_task(tmp_path) -> None:
    server, pipeline = _pipeline_server(tmp_path)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        code, body = _post(base + "/api/pipeline/run", {"task": "does-not-exist"})
        assert code == 400
        assert "unknown task" in body["error"]
        # a valid task id returns 200 ("started"); the child fails fast without
        # MT5 in the sandbox, which the supervisor reports, not the endpoint
        code, body = _post(base + "/api/pipeline/run", {"task": "collect"})
        assert code == 200
    finally:
        for tid in list(pipeline.running()):
            pipeline.stop(tid)
        server.stop()


def test_pipeline_logs_endpoint(tmp_path) -> None:
    server, pipeline = _pipeline_server(tmp_path)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        ok, _ = pipeline.run("learn", ["doctor"])
        assert ok
        deadline = time.time() + 10
        while not pipeline.running() and time.time() < deadline:
            time.sleep(0.05)
        code, body = _get(base + "/api/pipeline/logs?task=learn&lines=50")
        assert code == 200
        assert isinstance(json.loads(body), list)
        code, body = _get(base + "/api/pipeline/logs?task=nope&lines=50")
        assert code == 200 and json.loads(body) == []
    finally:
        server.stop()


def test_pipeline_task_defs_respect_settings() -> None:
    from slytrade.runtime.dashboard import pipeline_task_defs

    settings = {"symbols": ["EURUSD", "GBPUSD"], "timeframe": "H1", "lookback": "2y"}
    defs = pipeline_task_defs(settings)
    by_id = {d["id"]: d for d in defs}
    assert by_id["full-pipeline"]["argv"] == ["full-pipeline", "--symbol", "EURUSD", "--timeframe", "H1", "--lookback", "2y"]
    assert by_id["backtest"]["argv"] == ["persona-backtest", "--bars-file", "data/processed/aligned/EURUSD/h1/bars.parquet"]
    assert by_id["learn"]["argv"] == ["learn", "--bars-file", "data/processed/aligned/EURUSD/h1/bars.parquet"]
    assert by_id["collect"]["argv"] == ["collect-incremental", "--symbol", "EURUSD", "--lookback", "7d"]
    # the full watchlist reaches the admit stage
    assert by_id["admit"]["argv"] == ["admit", "--symbols", "EURUSD,GBPUSD", "--timeframe", "H1", "--lookback", "2y"]
