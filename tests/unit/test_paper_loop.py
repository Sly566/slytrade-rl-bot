from __future__ import annotations

import pandas as pd
import pytest

from slytrade.backtest.engine import BuyAndHoldOnceStrategy
from slytrade.runtime.paper_loop import BarBuilder, PaperTradingLoop, ReplayQuoteProvider
from slytrade.runtime.settings import RuntimeSettings
from slytrade.strategies.baselines import NoTradeStrategy


def make_ticks(minutes: int = 4, spread: float = 0.02, start: str = "2026-08-14T10:00:00") -> pd.DataFrame:
    times = pd.date_range(start, periods=minutes * 60, freq="1s", tz="UTC")
    mid = 100.0 + pd.Series(range(len(times)), dtype=float) * 0.0001
    bid = mid - spread / 2
    ask = mid + spread / 2
    return pd.DataFrame(
        {
            "time_msc": times,
            "symbol": "XAUUSD",
            "bid": bid.round(3),
            "ask": ask.round(3),
        }
    )


def _settings(tmp_path) -> RuntimeSettings:
    return RuntimeSettings(
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        kill_switch_path=str(tmp_path / "state" / "kill-switch.json"),
        json_logs=False,
        symbol="XAUUSD",
        timeframe="M1",
        poll_seconds=0.0,
    )


def test_bar_builder_buckets_ticks_into_bars() -> None:
    builder = BarBuilder("XAUUSD", "M1")
    completed = None
    ticks = make_ticks(minutes=2)
    for _, row in ticks.iterrows():
        from slytrade.backtest.execution import Quote

        quote = Quote(
            symbol="XAUUSD",
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            time=pd.Timestamp(row["time_msc"]).to_pydatetime(),
        )
        result = builder.on_quote(quote)
        if result is not None:
            completed = result
    final = builder.close_current()
    assert completed is not None or final is not None
    bars = [b for b in (completed, final) if b is not None]
    assert all(b["high"] >= b["low"] for b in bars)
    assert all("decision_time" in b for b in bars)


def test_paper_loop_no_trade(tmp_path) -> None:
    settings = _settings(tmp_path)
    provider = ReplayQuoteProvider(make_ticks(minutes=3), symbol="XAUUSD")
    loop = PaperTradingLoop(settings, provider, strategy=NoTradeStrategy())
    summary = loop.run()
    assert summary.bars_processed >= 2
    assert summary.orders_submitted == 0
    assert summary.errors == 0
    assert summary.final_equity == pytest.approx(settings.initial_balance, abs=0.01)


def test_paper_loop_entry_and_journal(tmp_path) -> None:
    settings = _settings(tmp_path)
    provider = ReplayQuoteProvider(make_ticks(minutes=4), symbol="XAUUSD")
    strategy = BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=0.1)
    loop = PaperTradingLoop(settings, provider, strategy=strategy)
    summary = loop.run()
    assert summary.bars_processed >= 3
    assert summary.orders_filled >= 1
    assert summary.errors == 0
    # Durable journal written for restart-safe rehydration.
    journal = tmp_path / "state" / "execution-events.db"
    assert journal.exists()
    # OMS/ledger can be rehydrated from the journal.
    from slytrade.execution.journal import SqliteJournal
    from slytrade.execution.ledger import TradeLedger
    from slytrade.execution.oms import OrderManagementSystem

    oms = OrderManagementSystem(SqliteJournal(journal))
    ledger = TradeLedger(SqliteJournal(journal))
    assert len(oms.orders) >= 1
    assert len(ledger.records) >= 1


def test_paper_loop_metrics_exposed(tmp_path) -> None:
    settings = _settings(tmp_path)
    provider = ReplayQuoteProvider(make_ticks(minutes=2), symbol="XAUUSD")
    loop = PaperTradingLoop(settings, provider, strategy=NoTradeStrategy())
    loop.run()
    # Metrics were exported to the Prometheus registry.
    sample = loop.metrics.registry.get_sample_value("slytrade_equity")
    assert sample is not None


def test_paper_loop_startup_blocked_on_bad_config(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.metrics_port = 0  # invalid
    provider = ReplayQuoteProvider(make_ticks(minutes=2), symbol="XAUUSD")
    with pytest.raises(ValueError, match="startup blocked"):
        PaperTradingLoop(settings, provider, strategy=NoTradeStrategy())


def test_news_gate_pauses_new_entries(tmp_path) -> None:
    from datetime import UTC, datetime

    from slytrade.runtime.news_gate import NewsEvent, NewsGate

    settings = _settings(tmp_path)
    provider = ReplayQuoteProvider(make_ticks(minutes=3), symbol="XAUUSD")
    loop = PaperTradingLoop(settings, provider, strategy=BuyAndHoldOnceStrategy(symbol="XAUUSD", volume=0.1))
    # An enabled gate covering the entire replay window pauses all new entries.
    loop.news_gate = NewsGate(
        enabled=True,
        events=(NewsEvent("COVER", datetime(2020, 1, 1, tzinfo=UTC), datetime(2035, 1, 1, tzinfo=UTC)),),
        quiet_before_minutes=0,
        quiet_after_minutes=0,
    )
    summary = loop.run()
    assert summary.orders_filled == 0
    sample = loop.metrics.registry.get_sample_value("slytrade_news_pauses_total")
    assert sample is not None and sample >= 1


def test_paper_loop_emits_stop_alert(tmp_path) -> None:
    from slytrade.runtime.alerting import Alert, AlertChannel, AlertManager

    class Recording(AlertChannel):
        def __init__(self) -> None:
            self.alerts: list[Alert] = []

        def send(self, alert: Alert) -> bool:
            self.alerts.append(alert)
            return True

    settings = _settings(tmp_path)
    provider = ReplayQuoteProvider(make_ticks(minutes=2), symbol="XAUUSD")
    loop = PaperTradingLoop(settings, provider, strategy=NoTradeStrategy())
    recording = Recording()
    loop.alerter = AlertManager([recording])
    loop.run()
    assert any(alert.title == "paper loop stopped" for alert in recording.alerts)
