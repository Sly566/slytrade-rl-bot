"""Live portfolio: one live loop per symbol, shared breaker, aggregated status."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd

from slytrade.cli import _resolve_portfolio_symbols
from slytrade.runtime.dashboard import build_loop_env, resolve_loop_command
from slytrade.runtime.demo_loop import LiveTradingLoop
from slytrade.runtime.live_portfolio import LivePortfolio
from slytrade.runtime.portfolio_loop import PortfolioBreaker
from slytrade.runtime.settings import RuntimeSettings, TradingStage


def _settings(tmp_path, **kw) -> RuntimeSettings:
    kwargs = dict(
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        data_dir=str(tmp_path / "data"),
        kill_switch_path=str(tmp_path / "state" / "kill-switch.json"),
        json_logs=False,
        symbol="XAUUSD",
        timeframe="M15",
        allow_live=True,
        stage=TradingStage.DEMO,
    )
    kwargs.update(kw)
    return RuntimeSettings(**kwargs)


class FakeMT5:
    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def symbols_get(self):
        return []

    def symbol_select(self, name, enable=True):
        return True

    def positions_get(self):
        return []

    def account_info(self):
        from types import SimpleNamespace

        return SimpleNamespace(equity=1000.0, balance=1000.0, currency="USD")


def test_resolve_loop_command_single_and_multi() -> None:
    assert resolve_loop_command({"loop_command": "live", "symbols": ["XAUUSD"]}) == "live"
    assert resolve_loop_command({"loop_command": "paper", "symbols": ["XAUUSD"]}) == "paper"
    # multi-symbol watchlist upgrades to the portfolio variants
    assert resolve_loop_command({"loop_command": "live", "symbols": ["XAUUSD", "EURUSD"]}) == "live-multi"
    assert resolve_loop_command({"loop_command": "paper", "symbols": ["XAUUSD", "EURUSD"]}) == "paper-multi"


def test_build_loop_env_includes_watchlist() -> None:
    env = build_loop_env({"symbols": ["XAUUSD", "USOIL", "BTCUSD"], "timeframe": "M15"}, {})
    assert env["SLYTRADE_SYMBOLS"] == "XAUUSD,USOIL,BTCUSD"
    assert env["SLYTRADE_SYMBOL"] == "XAUUSD"


def test_resolve_portfolio_symbols() -> None:
    settings = _settings(Path("/tmp/xx"))
    assert _resolve_portfolio_symbols("XAUUSD,USOIL", settings) == ["XAUUSD", "USOIL"]
    assert _resolve_portfolio_symbols("", settings) == ["XAUUSD"]


def test_portfolio_aggregates_per_symbol_status(tmp_path) -> None:
    settings = _settings(tmp_path)
    portfolio = LivePortfolio(["XAUUSD", "EURUSD"], settings)
    state = Path(settings.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    (state / "live_status_XAUUSD.json").write_text(json.dumps({"symbol": "XAUUSDm", "price": 4334.5, "side": "flat"}))
    (state / "live_status_EURUSD.json").write_text(json.dumps({"symbol": "EURUSDm", "price": 1.08, "side": "long"}))
    portfolio._publish_aggregate()
    agg = json.loads((state / "live_status.json").read_text())
    assert agg["mode"] == "portfolio"
    assert agg["count"] == 2
    assert set(agg["per_symbol"].keys()) == {"XAUUSD", "EURUSD"}
    assert agg["price"] == 4334.5  # first symbol's snapshot surfaced
    assert "XAUUSD" in agg["symbol"]


def test_journal_append_is_thread_safe(tmp_path) -> None:
    loop = LiveTradingLoop(_settings(tmp_path), FakeMT5())
    journal = Path(loop._journal_path())

    def write_rows(seed: int) -> None:
        for i in range(5):
            loop._journal_append({"time": f"2026-08-19T00:{seed:02d}:{i:02d}", "symbol": f"S{seed}", "outcome_r": float(seed + i)})

    threads = [threading.Thread(target=write_rows, args=(s,)) for s in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    frame = pd.read_parquet(journal)
    assert len(frame) == 100  # no lost/corrupted writes under concurrency


def test_journal_exit_records_into_portfolio_breaker(tmp_path) -> None:
    breaker = PortfolioBreaker(10_000.0, max_daily_drawdown=0.5, max_total_drawdown=0.5)
    loop = LiveTradingLoop(_settings(tmp_path), FakeMT5(), portfolio_breaker=breaker)
    loop._point_value = 100.0
    loop._journal_open = {
        "time": "2026-08-19T00:00:00", "symbol": "XAUUSDm", "side": "buy",
        "entry": 4000.0, "stop": 3997.0, "target": 4009.0, "volume": 0.1,
    }
    # bar hit the stop → realized = (3997 - 4000) * 100 * 0.1 = -30 USD
    from slytrade.backtest.execution import Quote

    quote = Quote(symbol="XAUUSDm", bid=3996.9, ask=3997.1, time=pd.Timestamp("2026-08-19T00:15:00").to_pydatetime())
    loop._journal_exit("XAUUSDm", {"high": 3998.0, "low": 3996.0}, quote)
    assert breaker._realized == -30.0
