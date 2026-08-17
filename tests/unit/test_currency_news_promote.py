"""Regression tests for currency wiring, the news gate, and the promotion gate."""
from __future__ import annotations

import json
from pathlib import Path

from slytrade.currency import CurrencyConverter, load_converter
from slytrade.runtime.news_gate import _recurring_events
from slytrade.runtime.settings import RuntimeSettings


def test_load_converter_reads_currency_block():
    conv = load_converter({"currency": {"rate_to_usd": 18.0}, "costs": {"currency_rate_to_usd": 1.0}})
    assert conv.fallback_rate == 18.0
    # Falls back to costs when the currency block is absent.
    conv2 = load_converter({"costs": {"currency_rate_to_usd": 15.0}})
    assert conv2.fallback_rate == 15.0


def test_converter_usd_is_noop():
    conv = CurrencyConverter(fallback_rate=18.0)
    conv.resolve(None, "USD")
    assert conv.rate == 1.0
    assert conv.to_usd(100.0) == 100.0


def test_recurring_events_include_nfp_fomc_cpi():
    events = _recurring_events(2026)
    names = {event.name for event in events}
    assert {"NFP", "FOMC", "CPI"}.issubset(names)
    # 12 NFP + 12 CPI + 8 FOMC = 32 windows.
    assert len(events) == 32


def test_settings_currency_defaults():
    settings = RuntimeSettings()
    assert settings.account_currency == "USD"
    assert settings.currency_rate_to_usd == 1.0
    assert settings.news_min_impact == "high"


def test_champion_baseline_roundtrip(tmp_path: Path, monkeypatch) -> None:
    import slytrade.tasks as tasks

    monkeypatch.setattr(tasks, "CHAMPION_BASELINE_PATH", tmp_path / "champion.json")
    tasks._write_champion_baseline("XAUUSD", 0.26799)
    baseline = tasks._load_champion_baseline()
    assert baseline is not None
    assert baseline["symbol"] == "XAUUSD"
    assert abs(baseline["total_return"] - 0.26799) < 1e-9


def test_promote_refuses_when_model_does_not_beat_champion(tmp_path: Path, monkeypatch) -> None:
    import slytrade.tasks as tasks

    artifacts = tmp_path / "artifacts"
    (artifacts / "model-1").mkdir(parents=True)
    (artifacts / "model-1" / "manifest.json").write_text(
        json.dumps({"model_id": "model-1", "metrics": {"mean_total_return": 0.05}}), encoding="utf-8"
    )
    monkeypatch.setattr(tasks, "CHAMPION_BASELINE_PATH", tmp_path / "champion.json")
    tasks._write_champion_baseline("XAUUSD", 0.26)

    result = tasks.promote(
        "model-1", stage="live", registry_path=str(tmp_path / "registry.jsonl"),
        artifacts_dir=str(artifacts), require_champion_beat=True,
    )
    assert not result.ok
    assert "does not beat" in result.message


def test_promote_allows_paper_without_baseline(tmp_path: Path, monkeypatch) -> None:
    # paper stage does not require the champion gate by default, so the missing
    # baseline must not block it. Stub the registry promote so we only assert
    # the gate behaviour.
    import slytrade.rl.deployment as deployment
    import slytrade.tasks as tasks

    monkeypatch.setattr(tasks, "CHAMPION_BASELINE_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(deployment, "promote_artifact", lambda model_id, **kwargs: {"model_id": model_id, "stage": kwargs.get("stage")})
    result = tasks.promote("model-x", stage="paper", registry_path=str(tmp_path / "registry.jsonl"))
    assert result.ok
    assert "champion baseline" not in result.message
