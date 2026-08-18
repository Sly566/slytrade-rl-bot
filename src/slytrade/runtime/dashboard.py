"""Web dashboard + control plane for the live bot.

A single, dependency-light HTTP server (stdlib ``ThreadingHTTPServer``) that
turns the trading loop into a platform you can watch and steer from a phone:

* ``GET  /``             — single-file mobile dashboard (no CDN, works offline)
* ``GET  /healthz``      — liveness (container health surface)
* ``GET  /readyz``       — readiness (loop state file present + fresh)
* ``GET  /api/status``   — the loop's latest snapshot (from state/live_status.json)
* ``GET  /api/trades``   — recent trades from the live journal (parquet)
* ``GET  /api/logs``     — tail of the structured log (or the supervised child)
* ``POST /api/control``  — start / stop / restart the supervised trading loop

Security: when ``SLYTRADE_DASHBOARD_TOKEN`` is set, every route requires a
Bearer token (header ``Authorization: Bearer …`` or ``?token=…``). Without a
token the dashboard binds for local use; **set a token before exposing it to
any network**. Pair with Tailscale / a reverse proxy for HTTPS.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


# --------------------------------------------------------------------------- #
# Loop supervisor (start/stop/restart the trading loop as a child process)
# --------------------------------------------------------------------------- #
@dataclass
class LoopSupervisor:
    command: list[str]
    cwd: str
    env: dict[str, str]
    _proc: subprocess.Popen | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _lines: deque = field(default_factory=lambda: deque(maxlen=5000), init=False)
    _started_at: str | None = field(default=None, init=False)
    _exit_code: int | None = field(default=None, init=False)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def exit_code(self) -> int | None:
        with self._lock:
            return self._exit_code

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return  # already running
            self._lines.clear()
            self._started_at = datetime.now(UTC).isoformat()
            self._exit_code = None
            self._proc = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        threading.Thread(target=self._pump, name="slytrade-supervisor", daemon=True).start()

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._lines.append(line.rstrip())
        code = proc.wait()
        with self._lock:
            self._exit_code = code

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()

    def tail(self, n: int) -> list[str]:
        with self._lock:
            lines = list(self._lines)
        return lines[-n:]

    def status(self) -> dict:
        return {
            "running": self.running,
            "pid": self._proc.pid if self._proc is not None else None,
            "started_at": self._started_at,
            "exit_code": self.exit_code,
            "command": " ".join(self.command),
        }


# --------------------------------------------------------------------------- #
# Dashboard server
# --------------------------------------------------------------------------- #
DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>SlyTrade Control</title>
<style>
:root{--bg:#0b0f14;--panel:#141a22;--panel2:#1b2430;--line:#243142;--txt:#dbe4ee;
--dim:#8fa1b5;--green:#2fbf71;--red:#e5534b;--amber:#e0a33d;--blue:#4aa3ff;--mono:ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;padding:14px;max-width:960px;margin:0 auto}
h1{font-size:18px;font-weight:600;display:flex;align-items:center;gap:8px;margin-bottom:4px}
.sub{color:var(--dim);font-size:12px;margin-bottom:14px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--dim);display:inline-block}
.dot.run{background:var(--green);box-shadow:0 0 8px var(--green)}
.dot.stop{background:var(--red)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.card .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:20px;font-weight:600;font-family:var(--mono);margin-top:3px;word-break:break-all}
.card .v.small{font-size:14px}
.banner{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--blue);border-radius:8px;padding:10px 12px;margin-bottom:14px;font-family:var(--mono);font-size:12.5px;color:var(--txt);white-space:pre-wrap}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.btn{background:var(--panel2);border:1px solid var(--line);color:var(--txt);border-radius:8px;padding:10px 16px;font-size:14px;cursor:pointer}
.btn.start{border-color:var(--green);color:var(--green)}
.btn.stop{border-color:var(--red);color:var(--red)}
.btn:disabled{opacity:.4;cursor:default}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-family:var(--mono);font-size:12px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:.05em}
td.win{color:var(--green)}td.loss{color:var(--red)}
.logs{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:10px;font-family:var(--mono);font-size:11.5px;height:280px;overflow-y:auto;white-space:pre-wrap;color:#b7c4d1}
.section{margin:16px 0 6px;color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.08em}
input[type=password]{background:var(--panel2);border:1px solid var(--line);color:var(--txt);border-radius:8px;padding:10px;font-size:14px;width:100%;margin-bottom:10px}
#login{max-width:340px;margin:40px auto}
</style>
</head>
<body>
<div id="login" style="display:none">
  <h1>🔐 SlyTrade Control</h1>
  <div class="sub">Enter your access token</div>
  <input type="password" id="token" placeholder="Access token"/>
  <button class="btn start" onclick="saveToken()" style="width:100%">Unlock</button>
</div>
<div id="app">
  <h1><span class="dot" id="dot"></span> SlyTrade Control</h1>
  <div class="sub" id="subtitle">connecting…</div>
  <div class="banner" id="decision">— waiting for the loop —</div>
  <div class="grid">
    <div class="card"><div class="k">Symbol</div><div class="v small" id="symbol">—</div></div>
    <div class="card"><div class="k">Price</div><div class="v" id="price">—</div></div>
    <div class="card"><div class="k">Equity</div><div class="v" id="equity">—</div></div>
    <div class="card"><div class="k">Balance</div><div class="v" id="balance">—</div></div>
    <div class="card"><div class="k">Position</div><div class="v small" id="side">—</div></div>
    <div class="card"><div class="k">Pending limit</div><div class="v small" id="pending">—</div></div>
    <div class="card"><div class="k">Bars built</div><div class="v" id="bars">—</div></div>
    <div class="card"><div class="k">Ticks / errors</div><div class="v small" id="ticks">—</div></div>
  </div>
  <div class="section">Control</div>
  <div class="row">
    <button class="btn start" id="btnStart" onclick="ctl('start')">▶ Start loop</button>
    <button class="btn stop" id="btnStop" onclick="ctl('stop')">■ Stop loop</button>
    <button class="btn" onclick="ctl('restart')">↻ Restart</button>
    <span class="sub" id="ctlstate"></span>
  </div>
  <div class="section">Recent trades</div>
  <table><thead><tr><th>time</th><th>side</th><th>entry</th><th>exit</th><th>R</th><th>reason</th></tr></thead>
  <tbody id="trades"><tr><td colspan="6">no trades yet</td></tr></tbody></table>
  <div class="section">Log tail</div>
  <div class="logs" id="logs">—</div>
</div>
<script>
var TOKEN = localStorage.getItem("slytrade_token") || "";
function authHeaders(){ return TOKEN ? {"Authorization":"Bearer "+TOKEN} : {}; }
function needsLogin(){ var e=document.getElementById("login"); var a=document.getElementById("app");
  var locked = e && e.style.display!=="none"; a.style.display = locked?"none":"block"; }
function saveToken(){ TOKEN=document.getElementById("token").value; localStorage.setItem("slytrade_token",TOKEN); refresh(); }
async function api(path, opts){
  opts = opts || {}; opts.headers = Object.assign({}, authHeaders(), opts.headers||{});
  var r = await fetch(path, opts);
  if(r.status===401){ document.getElementById("login").style.display="block"; document.getElementById("app").style.display="none"; throw new Error("unauthorized"); }
  document.getElementById("login").style.display="none"; document.getElementById("app").style.display="block";
  return r.json();
}
function ctl(action){
  api("/api/control", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({action:action})})
    .then(d=>{ document.getElementById("ctlstate").textContent = d.detail||""; refresh(); })
    .catch(()=>{});
}
async function refresh(){
  var s; try{ s = await api("/api/status"); }catch(e){ return; }
  var running = s.loop && s.loop.running;
  var dot=document.getElementById("dot"); dot.className = "dot "+(running?"run":"stop");
  document.getElementById("subtitle").textContent = s.loop ? ("supervisor: "+(running?"RUNNING":"stopped")+" · "+s.loop.command) : "no loop";
  if(s.status){
    document.getElementById("symbol").textContent = (s.status.symbol||"—")+" "+(s.status.timeframe||"");
    document.getElementById("price").textContent = s.status.price!=null? s.status.price : "—";
    document.getElementById("equity").textContent = s.status.equity!=null? s.status.equity : "—";
    document.getElementById("balance").textContent = s.status.balance!=null? s.status.balance : "—";
    document.getElementById("side").textContent = s.status.side||"—";
    var p=s.status.pending_limit; document.getElementById("pending").textContent = p? (p.side+" @"+p.price+" ("+p.bars+" bars)") : "—";
    document.getElementById("bars").textContent = s.status.bars_built||0;
    document.getElementById("ticks").textContent = (s.status.tick||0)+" / "+(s.status.errors||0);
    document.getElementById("decision").textContent = s.status.last_decision || "— waiting for the first bar —";
  }
  document.getElementById("btnStart").disabled = running;
  document.getElementById("btnStop").disabled = !running;
  try{
    var t = await api("/api/trades"); var tb=document.getElementById("trades");
    if(t && t.length){ tb.innerHTML = t.map(function(x){ return "<tr><td>"+x.time+"</td><td>"+x.side+"</td><td>"+x.entry+"</td><td>"+(x.exit||"")+"</td><td class='"+(x.outcome_r>0?"win":"loss")+"'>"+(x.outcome_r!=null?x.outcome_r:"")+"</td><td>"+(x.exit_reason||"")+"</td></tr>"; }).join(""); }
  }catch(e){}
  try{
    var l = await api("/api/logs?lines=80"); var el=document.getElementById("logs");
    if(l && l.length){ el.textContent = l.join("\n"); el.scrollTop = el.scrollHeight; }
  }catch(e){}
}
refresh(); setInterval(refresh, 3000);
</script>
</body>
</html>"""


@dataclass
class DashboardServer:
    host: str = "0.0.0.0"
    port: int = 8080
    token: str = ""
    state_dir: str = "state"
    data_dir: str = "data"
    log_dir: str = "logs"
    supervisor: LoopSupervisor | None = None
    _server: ThreadingHTTPServer | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    # -- helpers -------------------------------------------------------------
    def _status_file(self) -> Path:
        return Path(self.state_dir) / "live_status.json"

    def _journal_file(self) -> Path:
        return Path(self.data_dir) / "live_journal" / "trades.parquet"

    def _log_file(self) -> Path:
        return Path(self.log_dir) / "slytrade.jsonl"

    def read_status(self) -> dict | None:
        path = self._status_file()
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def read_trades(self, limit: int = 200) -> list[dict]:
        path = self._journal_file()
        if not path.exists():
            return []
        try:
            import pandas as pd

            frame = pd.read_parquet(path).tail(limit)
            out = []
            for _, row in frame.iterrows():
                out.append({
                    "time": str(row.get("time", "")),
                    "side": row.get("side", ""),
                    "entry": _num(row.get("entry")),
                    "exit": _num(row.get("exit")),
                    "outcome_r": _num(row.get("outcome_r")),
                    "exit_reason": str(row.get("exit_reason", "")),
                    "volume": _num(row.get("volume")),
                })
            return out
        except Exception:  # pragma: no cover
            return []

    def read_logs(self, lines: int = 100) -> list[str]:
        if self.supervisor is not None and self.supervisor.running:
            return self.supervisor.tail(lines)
        path = self._log_file()
        if not path.exists():
            return []
        try:
            with path.open("r", errors="replace") as handle:
                tail = handle.readlines()[-lines:]
            cleaned = []
            for line in tail:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ts = obj.get("asctime") or obj.get("timestamp") or ""
                    level = obj.get("levelname") or ""
                    msg = obj.get("message") or obj.get("msg") or ""
                    cleaned.append(f"{ts} {level} {msg}".strip())
                except Exception:
                    cleaned.append(line)
            return cleaned
        except Exception:  # pragma: no cover
            return []

    # -- HTTP ----------------------------------------------------------------
    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                return

            def _authorized(self) -> bool:
                # ``server.token`` may hold one token or a comma-separated list
                # of tokens (one per user/device). Empty = open (localhost use).
                tokens = {t.strip() for t in server.token.split(",") if t.strip()}
                if not tokens:
                    return True
                header = self.headers.get("Authorization", "")
                if header.startswith("Bearer "):
                    return header[7:].strip() in tokens
                query = parse_qs(urlparse(self.path).query)
                return query.get("token", [""])[0] in tokens

            def _send(self, code: int, body: bytes, ctype: str, *, cache_control: str | None = None) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                if cache_control:
                    self.send_header("Cache-Control", cache_control)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, payload: object) -> None:
                self._send(code, json.dumps(payload, default=str).encode(), "application/json; charset=utf-8", cache_control="no-store")

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/healthz":
                    self._send(200, b"ok\n", "text/plain")
                    return
                if path == "/readyz":
                    status = server.read_status()
                    if status is None:
                        self._send(503, b"not ready: no loop status yet\n", "text/plain")
                        return
                    self._send(200, b"ready\n", "text/plain")
                    return
                if not self._authorized():
                    if path == "/":
                        self._send(401, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
                    else:
                        self._json(401, {"error": "unauthorized"})
                    return
                if path == "/":
                    self._send(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8", cache_control="no-store")
                    return
                if path == "/api/status":
                    self._json(200, {
                        "loop": server.supervisor.status() if server.supervisor else None,
                        "status": server.read_status(),
                    })
                    return
                if path == "/api/trades":
                    self._json(200, server.read_trades())
                    return
                if path == "/api/logs":
                    query = parse_qs(urlparse(self.path).query)
                    lines = int(query.get("lines", ["100"])[0] or 100)
                    self._json(200, server.read_logs(min(max(lines, 1), 2000)))
                    return
                self._json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path != "/api/control":
                    self._json(404, {"error": "not found"})
                    return
                if not self._authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except Exception:
                    body = {}
                action = str(body.get("action", "")).lower()
                if server.supervisor is None:
                    self._json(400, {"error": "no supervised loop configured"})
                    return
                if action == "start":
                    server.supervisor.start()
                    self._json(200, {"action": "start", "detail": "loop starting"})
                elif action == "stop":
                    server.supervisor.stop()
                    self._json(200, {"action": "stop", "detail": "loop stopped"})
                elif action == "restart":
                    server.supervisor.stop()
                    time.sleep(0.3)
                    server.supervisor.start()
                    self._json(200, {"action": "restart", "detail": "loop restarting"})
                else:
                    self._json(400, {"error": f"unknown action {action!r}"})

        return Handler

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        self._server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, name="slytrade-dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def bound_port(self) -> int:
        if self._server is None:
            return self.port
        return int(self._server.server_address[1])


def _num(value: Any):
    try:
        result = float(value)
        return result if result == result else None  # NaN -> None
    except Exception:
        return None


def run_dashboard(
    *,
    host: str,
    port: int,
    command: str,
    token: str,
    supervise: bool,
    state_dir: str = "state",
    data_dir: str = "data",
    log_dir: str = "logs",
) -> DashboardServer:
    """Start the dashboard (and optionally the supervised loop) and block.

    Returns the running server; call ``stop()`` to shut it down. When
    ``supervise`` is True the trading loop is spawned as a child process and
    the /api/control endpoints manage it.
    """
    env = dict(os.environ)
    supervisor = None
    if supervise:
        supervisor = LoopSupervisor(
            command=[sys.executable, "-m", "slytrade.cli", command],
            cwd=os.getcwd(),
            env=env,
        )
        supervisor.start()
    server = DashboardServer(
        host=host,
        port=port,
        token=token,
        state_dir=state_dir,
        data_dir=data_dir,
        log_dir=log_dir,
        supervisor=supervisor,
    )
    server.start()
    return server
