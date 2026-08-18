"""Visibility features: warmup logging + per-bar decision trace + heartbeat.

The user must be able to SEE the live loop working at every step, so these
guard the instrumentation that makes silence impossible.
"""
from __future__ import annotations

import pandas as pd

from slytrade.runtime.demo_loop import LiveTradingLoop
from slytrade.runtime.settings import RuntimeSettings, TradingStage


class FakeMT5:
    def __init__(self) -> None:
        self.initialized = True

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
    )


def _loop(tmp_path) -> LiveTradingLoop:
    return LiveTradingLoop(_settings(tmp_path), FakeMT5())


def _m15_bars(minutes: int = 24 * 60) -> pd.DataFrame:
    times = pd.date_range("2026-08-01T00:00:00", periods=minutes // 15, freq="15min", tz="UTC")
    close = 4000.0 + pd.Series(range(len(times)), dtype=float) * 0.05
    return pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",  # canonical, NOT the broker suffix
            "timeframe": "M15",
            "open": close - 0.2, "high": close + 0.3, "low": close - 0.4, "close": close,
            "tick_volume": 100.0, "spread": 0.135,
        }
    )


def test_warmup_matches_canonical_symbol(tmp_path) -> None:
    bars = _m15_bars()
    warm_file = tmp_path / "warm.parquet"
    bars.to_parquet(warm_file)
    loop = _loop(tmp_path)
    loop.settings.replay_bars_file = str(warm_file)
    loop._warmup("XAUUSDm")  # broker suffix must still match the canonical column
    assert len(loop._window_bars) == len(bars)


def test_warmup_logs_when_no_file_configured(tmp_path, caplog) -> None:
    import logging

    caplog.set_level(logging.INFO)
    loop = _loop(tmp_path)
    loop.logger.propagate = True  # route into caplog (setup_logging sets propagate=False)
    loop._warmup("XAUUSDm")
    assert any("no replay bars file" in r.message for r in caplog.records)


def test_warmup_auto_discovers_pipeline_output(tmp_path, caplog) -> None:
    """Running the live loop right after the full pipeline must warm up without
    any SLYTRADE_REPLAY_BARS_FILE — the aligned output is auto-discovered."""
    import logging

    caplog.set_level(logging.INFO)
    bars = _m15_bars()
    aligned_dir = tmp_path / "data" / "processed" / "aligned" / "XAUUSD" / "m15"
    aligned_dir.mkdir(parents=True)
    bars.to_parquet(aligned_dir / "bars.parquet")

    loop = _loop(tmp_path)
    loop.logger.propagate = True
    loop.settings.data_dir = str(tmp_path / "data")
    loop.settings.replay_bars_file = None  # nothing configured
    loop._warmup("XAUUSDm")
    assert len(loop._window_bars) == len(bars)
    assert any("auto-discovered" in r.message for r in caplog.records)
    assert any("warmup loaded" in r.message for r in caplog.records)


def test_decision_trace_hold_and_signal(tmp_path) -> None:
    loop = _loop(tmp_path)
    series = pd.Series(
        {
            "time": pd.Timestamp("2026-08-18T12:00:00", tz="UTC"), "symbol": "XAUUSD", "timeframe": "M15",
            "open": 4000.0, "high": 4000.5, "low": 3999.5, "close": 4000.0,
            "atr": 3.0, "atr_norm": 0.001, "trend_strength": 0.0, "premium_discount": 0.0,
            "bos_dir": 0.0, "choch_dir": 0.0, "liquidity_sweep": 0.0, "fvg_bullish": 0.0, "fvg_bearish": 0.0,
            "order_block_bullish": 0.0, "order_block_bearish": 0.0,
            "htf_h4_bos_dir": 0.0, "mtf_bias": 0.0, "mtf_confluence_score": 0.0,
        }
    )
    from slytrade.backtest.execution import Quote

    quote = Quote(symbol="XAUUSDm", bid=3999.9, ask=4000.1, time=series["time"].to_pydatetime())
    hold = loop._decision_trace(series, quote, None)
    assert "HOLD" in hold
    assert "long=" in hold and "short=" in hold

    from slytrade.execution.models import OrderIntent, OrderKind, Side

    intent = OrderIntent(symbol="XAUUSDm", side=Side.BUY, volume=0.1, kind=OrderKind.LIMIT, limit_price=3999.25)
    signal = loop._decision_trace(series, quote, intent)
    assert "SIGNAL" in signal
    assert "buy" in signal and "limit" in signal


def test_status_heartbeat_respects_interval(tmp_path, caplog) -> None:
    import logging

    caplog.set_level(logging.INFO)
    loop = _loop(tmp_path)
    loop.logger.propagate = True
    from slytrade.backtest.execution import Quote

    quote = Quote(symbol="XAUUSDm", bid=3999.9, ask=4000.1, time=pd.Timestamp("2026-08-18T12:00:00").to_pydatetime())
    loop._status_heartbeat(quote, "XAUUSDm")  # first call: logs
    n1 = sum("alive" in r.message for r in caplog.records)
    loop._status_heartbeat(quote, "XAUUSDm")  # inside interval: no new log
    n2 = sum("alive" in r.message for r in caplog.records)
    assert n1 == 1 and n2 == 1
    loop._last_heartbeat_at = 0.0  # force expiry
    loop._status_heartbeat(quote, "XAUUSDm")
    n3 = sum("alive" in r.message for r in caplog.records)
    assert n3 == 2
    # The console message carries the details (price/equity/state) inline.
    assert "price=" in caplog.records[-1].message
