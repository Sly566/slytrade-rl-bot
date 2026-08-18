"""Broker warmup: the bot joins the market CURRENT, not cold or stale.

The warmup must seed the feature window from the terminal's own recent bars
(ending "now", current price level) and CONTINUE the in-progress bar instead of
rebuilding it from scratch — so there is no multi-hour wait and no stale price
level to un-learn.
"""
from __future__ import annotations

import logging

import pandas as pd

from slytrade.backtest.execution import Quote
from slytrade.runtime.demo_loop import LiveTradingLoop
from slytrade.runtime.paper_loop import BarBuilder
from slytrade.runtime.settings import RuntimeSettings, TradingStage


def _bar_rows(n: int, *, start_minutes_ago: int = 2000, step_minutes: int = 15) -> list[dict]:
    """MT5-style rate rows (time in seconds) ending at 'now'."""
    end = pd.Timestamp("2026-08-18T01:15:00", tz="UTC")
    rows = []
    for i in range(n):
        ts = end - pd.Timedelta(minutes=(n - 1 - i) * step_minutes)
        price = 4334.0 + i * 0.05
        rows.append(
            {
                "time": int(ts.timestamp()),
                "open": price, "high": price + 0.3, "low": price - 0.4, "close": price,
                "tick_volume": 100, "spread": 0.135, "real_volume": 0.0,
            }
        )
    return rows


class FakeMT5:
    TIMEFRAME_M15 = 15

    def __init__(self) -> None:
        self.rows = _bar_rows(50)

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

        return SimpleNamespace(equity=1136.34, balance=1000.0)

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int):
        return self.rows


def _settings(tmp_path) -> RuntimeSettings:
    return RuntimeSettings(
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        kill_switch_path=str(tmp_path / "state" / "kill-switch.json"),
        json_logs=False,
        symbol="XAUUSD",
        timeframe="M15",
        poll_seconds=0.01,
        allow_live=True,
        stage=TradingStage.DEMO,
        history_bars=50,
    )


def test_bar_builder_seed_continues_current_bar() -> None:
    builder = BarBuilder("XAUUSDm", "M15")
    builder.seed(
        {
            "time": pd.Timestamp("2026-08-18T01:00:00", tz="UTC"),
            "symbol": "XAUUSDm", "timeframe": "M15",
            "open": 4334.0, "high": 4335.0, "low": 4333.0, "close": 4334.5,
            "tick_volume": 50, "spread": 0.135,
        }
    )
    # A quote still inside the seeded bucket updates it WITHOUT completing.
    same = Quote(symbol="XAUUSDm", bid=4334.4, ask=4334.6,
                 time=pd.Timestamp("2026-08-18T01:10:00", tz="UTC").to_pydatetime())
    assert builder.on_quote(same) is None
    # A quote in the NEXT bucket completes the seeded bar at its open time.
    nxt = Quote(symbol="XAUUSDm", bid=4334.8, ask=4335.0,
                time=pd.Timestamp("2026-08-18T01:15:30", tz="UTC").to_pydatetime())
    completed = builder.on_quote(nxt)
    assert completed is not None
    assert completed["time"] == pd.Timestamp("2026-08-18T01:00:00", tz="UTC")
    assert completed["tick_volume"] >= 51  # 50 seeded + the mid-bar tick
    assert completed["high"] == 4335.0  # seeded high preserved


def test_broker_warmup_seeds_window_and_continues_bar(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO)
    loop = LiveTradingLoop(_settings(tmp_path), FakeMT5())
    loop.logger.propagate = True
    loop._broker_warmup("XAUUSDm")
    # 49 completed bars seed the feature window; the 50th (in-progress) seeds the builder.
    assert len(loop._window_bars) == 49
    assert loop.bar_builder._current is not None
    assert any("broker warmup" in r.message for r in caplog.records)
    assert any("continuing the current bar" in r.message for r in caplog.records)


def test_broker_warmup_falls_back_to_file(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO)
    mt5 = FakeMT5()
    mt5.copy_rates_from_pos = lambda *a, **k: None  # terminal gives nothing
    loop = LiveTradingLoop(_settings(tmp_path), mt5)
    loop.logger.propagate = True
    # file warmup with no file -> the clear cold-start hint, never a crash
    loop._broker_warmup("XAUUSDm")
    assert any("falling back to file warmup" in r.message or "no replay bars file" in r.message for r in caplog.records)
