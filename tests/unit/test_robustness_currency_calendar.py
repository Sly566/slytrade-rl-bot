"""Tests for robustness, currency conversion and the calendar feed."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from slytrade.currency import CurrencyConverter
from slytrade.rl.robustness import monte_carlo_trades, perturbation_sweep, regime_segmentation
from slytrade.runtime.calendar import calendar_gate, load_calendar_entries


# --- Monte Carlo -----------------------------------------------------------
def test_monte_carlo_reports_loss_probability() -> None:
    # A net-losing trade sequence should report high P(loss).
    report = monte_carlo_trades([-10.0, -5.0, -7.0, 3.0, -4.0], n_simulations=500, seed=42)
    assert report.observed_total < 0
    assert 0.0 <= report.prob_loss <= 1.0
    assert report.ci_95_low <= report.ci_95_high
    assert report.worst_total <= report.best_total


def test_monte_carlo_requires_trades() -> None:
    with pytest.raises(ValueError):
        monte_carlo_trades([])


# --- Perturbation -----------------------------------------------------------
def test_perturbation_sweep_reports_sensitivity() -> None:
    def score(params):
        return params["stop_loss_atr"] * 10.0

    results = perturbation_sweep(score, {"stop_loss_atr": 1.0}, deltas={"stop_loss_atr": (-0.5, 0.5)})
    assert len(results) == 1
    assert results[0].param == "stop_loss_atr"
    assert results[0].spread > 0
    assert results[0].sensitive


# --- Regime segmentation ----------------------------------------------------
def test_regime_segmentation_by_session() -> None:
    times = pd.date_range("2026-08-14T08:00:00", periods=60, freq="min", tz="UTC")
    bars = pd.DataFrame({"time": times, "close": 100.0})
    trades = pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-08-14T08:05:00", tz="UTC"), pd.Timestamp("2026-08-14T08:10:00", tz="UTC")],
            "realized_pnl": [10.0, -5.0],
        }
    )
    segments = regime_segmentation(trades, bars)
    assert len(segments) >= 1
    assert sum(segment.trades for segment in segments) == 2


# --- Currency ---------------------------------------------------------------
class FakeTick:
    def __init__(self, bid, ask):
        self.bid = bid
        self.ask = ask


class FakeMT5:
    def __init__(self):
        self.ticks = {"USDZAR": FakeTick(18.0, 18.2)}

    def symbol_info_tick(self, symbol):
        return self.ticks.get(symbol)


def test_currency_usd_passthrough() -> None:
    converter = CurrencyConverter(fallback_rate=18.0)
    assert converter.resolve(FakeMT5(), "USD") == 1.0
    assert converter.to_usd(1000.0, FakeMT5(), "USD") == 1000.0


def test_currency_usdzar_conversion() -> None:
    converter = CurrencyConverter(fallback_rate=20.0)
    rate = converter.resolve(FakeMT5(), "ZAR")
    # USDZAR mid ≈ 18.1 → 1 ZAR ≈ 0.0552 USD
    assert 0.05 < rate < 0.06
    assert converter.to_usd(1000.0, FakeMT5(), "ZAR") == pytest.approx(1000.0 * rate)


def test_currency_fallback_when_no_pair() -> None:
    converter = CurrencyConverter(fallback_rate=17.5)
    assert converter.resolve(FakeMT5(), "AED") == 17.5


# --- Calendar feed ----------------------------------------------------------
def test_calendar_gate_from_json_file(tmp_path: Path) -> None:
    path = tmp_path / "calendar.json"
    path.write_text(
        json.dumps(
            {
                "events": [
                    {"name": "NFP", "start_utc": "2026-09-04T12:30:00", "end_utc": "2026-09-04T13:45:00", "impact": "high"},
                    {"name": "Minor", "start_utc": "2026-09-04T09:00:00", "end_utc": "2026-09-04T09:30:00", "impact": "low"},
                ]
            }
        ),
        encoding="utf-8",
    )
    gate = calendar_gate(path=str(path), min_impact="high", quiet_before_minutes=0, quiet_after_minutes=0)
    assert gate.enabled
    assert gate.is_red_folder(datetime(2026, 9, 4, 13, 0, tzinfo=UTC))
    # Low-impact event filtered out.
    assert not gate.is_red_folder(datetime(2026, 9, 4, 9, 15, tzinfo=UTC))


def test_calendar_gate_disabled_without_source() -> None:
    gate = calendar_gate()
    assert not gate.enabled


def test_calendar_entries_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"
    path.write_text(
        "name,start_utc,end_utc,impact\nCPI,2026-09-10T12:30:00,2026-09-10T13:30:00,high\n",
        encoding="utf-8",
    )
    entries = load_calendar_entries(path=str(path), min_impact="high")
    assert len(entries) == 1
    assert entries[0].name == "CPI"
