"""Dashboard settings store + endpoints: the operator's configuration surface."""
from __future__ import annotations

import json

from slytrade.runtime.dashboard import DashboardServer, build_loop_env
from slytrade.runtime.settings_store import (
    DashboardSettings,
    default_dashboard_settings,
    load_dashboard_settings,
    save_dashboard_settings,
)


def test_defaults_come_from_env() -> None:
    s = default_dashboard_settings({"SLYTRADE_SYMBOL": "EURUSD", "SLYTRADE_TIMEFRAME": "H1"})
    assert s.symbols == ["EURUSD"]
    assert s.timeframe == "H1"


def test_save_and_reload_roundtrip(tmp_path) -> None:
    path = tmp_path / "dashboard_settings.json"
    saved, problems = save_dashboard_settings(path, {"symbols": ["XAUUSD", "EURUSD"], "timeframe": "H4", "risk_per_trade": 0.01}, {})
    assert problems == []
    assert saved.symbols == ["XAUUSD", "EURUSD"]
    assert saved.timeframe == "H4"
    assert saved.risk_per_trade == 0.01
    reloaded = load_dashboard_settings(path, {})
    assert reloaded.to_dict() == saved.to_dict()


def test_save_rejects_invalid_and_keeps_previous(tmp_path) -> None:
    path = tmp_path / "dashboard_settings.json"
    save_dashboard_settings(path, {"symbols": ["XAUUSD"]}, {})
    bad, problems = save_dashboard_settings(path, {"risk_per_trade": 0.5, "timeframe": "XXL"}, {})
    assert problems  # non-empty
    assert any("risk_per_trade" in p for p in problems)
    assert any("timeframe" in p for p in problems)
    # invalid save did NOT persist — previous settings still load
    assert load_dashboard_settings(path, {}).risk_per_trade == 0.005


def test_validate_rejects_bad_values() -> None:
    assert DashboardSettings(symbols=[]).validate()
    assert DashboardSettings(symbols=["XAUUSD"], timeframe="ZZ").validate()
    assert DashboardSettings(symbols=["XAUUSD"], risk_per_trade=0.5).validate()
    assert DashboardSettings(symbols=["XAUUSD"], max_position_volume=0).validate()
    assert DashboardSettings(symbols=["XAUUSD"], limit_entry_atr=-1).validate()
    assert not DashboardSettings(symbols=["XAUUSD"], timeframe="M15").validate()


def test_build_loop_env_maps_settings() -> None:
    env = build_loop_env(
        {"symbols": ["EURUSD"], "timeframe": "H1", "risk_per_trade": 0.01, "max_position_volume": 2.5, "limit_entry_atr": 0.5},
        {},
    )
    assert env["SLYTRADE_SYMBOL"] == "EURUSD"
    assert env["SLYTRADE_TIMEFRAME"] == "H1"
    assert env["SLYTRADE_RISK_PER_TRADE"] == "0.01"
    assert env["SLYTRADE_MAX_POSITION_VOLUME"] == "2.5"
    assert env["SLYTRADE_LIMIT_ENTRY_ATR"] == "0.5"


def _get(url, token=None):
    import urllib.error
    import urllib.request

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url, payload, token=None):
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_settings_endpoints_get_and_save(tmp_path) -> None:
    spath = tmp_path / "dashboard_settings.json"
    server = DashboardServer(
        host="127.0.0.1", port=0,
        state_dir=str(tmp_path), data_dir=str(tmp_path), log_dir=str(tmp_path),
        settings_path=str(spath),
        settings=default_dashboard_settings({}).to_dict(),
        base_env={},
    )
    server.start()
    try:
        base = f"http://127.0.0.1:{server.bound_port}"
        code, body = _get(base + "/api/settings")
        assert code == 200
        payload = json.loads(body)
        assert payload["settings"]["symbols"] == ["XAUUSD"]
        assert "M15" in payload["timeframes"]

        code, body = _post(base + "/api/settings", {"symbols": ["XAUUSD", "EURUSD"], "timeframe": "H1", "risk_per_trade": 0.01})
        assert code == 200
        assert body["settings"]["symbols"] == ["XAUUSD", "EURUSD"]
        # persisted to disk
        assert json.loads(spath.read_text())["timeframe"] == "H1"

        # invalid save -> 400 with problems, unchanged
        code, body = _post(base + "/api/settings", {"risk_per_trade": 9.0})
        assert code == 400
        assert body["problems"]
        assert json.loads(spath.read_text())["risk_per_trade"] == 0.01
    finally:
        server.stop()
