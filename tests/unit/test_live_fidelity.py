"""Live-loop fidelity: MTF (H4/D1) context, warmup, and the portfolio breaker.

These guard the fix that makes the LIVE paper loop trade the same strategy the
backtest validates: the champion's H4-trend alignment gate must actually fire
live (it was silently skipped when the loop only computed single-timeframe
features), and a multi-symbol book must halt on aggregate drawdown.
"""
from __future__ import annotations

import pandas as pd

from slytrade.data.resample import resample_bars_to_timeframe
from slytrade.runtime.paper_loop import PaperTradingLoop, ReplayQuoteProvider
from slytrade.runtime.portfolio_loop import PortfolioBreaker
from slytrade.runtime.settings import RuntimeSettings
from slytrade.strategies.baselines import NoTradeStrategy


def make_m15_bars(minutes: int = 4 * 24 * 60, start: str = "2026-08-01T00:00:00") -> pd.DataFrame:
    times = pd.date_range(start, periods=minutes // 15, freq="15min", tz="UTC")
    n = len(times)
    close = 4000.0 + pd.Series(range(n), dtype=float) * 0.05
    return pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",
            "timeframe": "M15",
            "open": close - 0.2,
            "high": close + 0.3,
            "low": close - 0.4,
            "close": close,
            "tick_volume": 100.0,
            "spread": 0.135,
        }
    )


def _ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_msc": pd.date_range("2026-08-01T00:00:00", periods=1, freq="1s", tz="UTC"),
            "symbol": "XAUUSD",
            "bid": [3999.9],
            "ask": [4000.1],
        }
    )


def _settings(tmp_path, timeframe: str = "M15", history_bars: int = 5000) -> RuntimeSettings:
    return RuntimeSettings(
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
        kill_switch_path=str(tmp_path / "state" / "kill-switch.json"),
        json_logs=False,
        symbol="XAUUSD",
        timeframe=timeframe,
        poll_seconds=0.0,
        history_bars=history_bars,
    )


def test_resample_bars_to_timeframe_ohlc_and_grid() -> None:
    bars = make_m15_bars()
    h4 = resample_bars_to_timeframe(bars, "H4")
    d1 = resample_bars_to_timeframe(bars, "D1")
    assert 0 < len(h4) < len(bars)
    for column in ("time", "symbol", "open", "high", "low", "close", "tick_volume", "spread"):
        assert column in h4.columns
    assert (h4["high"] >= h4["low"]).all()
    assert (h4["high"] >= h4["open"]).all() and (h4["high"] >= h4["close"]).all()
    # H4 bars land on the 4-hour grid (0, 4, 8, ... UTC).
    assert (h4["time"].dt.hour % 4 == 0).all()
    assert len(d1) >= 1
    # Bar-open timestamps: the first H4 open is the first M15 bar's open.
    assert h4["time"].iloc[0] == bars["time"].iloc[0]


def test_paper_loop_decision_bar_carries_mtf_context(tmp_path) -> None:
    settings = _settings(tmp_path)
    loop = PaperTradingLoop(settings, ReplayQuoteProvider(_ticks(), symbol="XAUUSD"), strategy=NoTradeStrategy())
    loop._window_bars = [row.to_dict() for _, row in make_m15_bars().iterrows()]
    decision = loop._decision_bar(loop._window_bars[-1])
    assert "mtf_bias" in decision.index
    assert "mtf_confluence_score" in decision.index
    assert any(column.startswith("htf_") for column in decision.index)
    # The higher-timeframe columns are finite scalars.
    assert pd.notna(decision["mtf_bias"]) and pd.notna(decision["mtf_confluence_score"])


def test_paper_loop_warmup_seeds_window(tmp_path) -> None:
    bars = make_m15_bars(minutes=8 * 60)  # 32 bars
    warm_file = tmp_path / "warm.parquet"
    bars.to_parquet(warm_file)
    settings = _settings(tmp_path)
    settings.replay_bars_file = str(warm_file)
    loop = PaperTradingLoop(settings, ReplayQuoteProvider(_ticks(), symbol="XAUUSD"), strategy=NoTradeStrategy())
    assert loop._window_bars == []
    loop._warmup()
    assert len(loop._window_bars) == len(bars)


def test_portfolio_breaker_trips_on_total_drawdown() -> None:
    breaker = PortfolioBreaker(100_000.0, max_total_drawdown=0.08, max_daily_drawdown=0.20)
    assert breaker.allowed()
    breaker.record("XAUUSD", -5_000.0)
    assert breaker.allowed()  # 5% total dd, under both limits
    breaker.record("EURUSD", -4_000.0)
    assert not breaker.allowed()  # 9% total dd > 8%
    assert breaker.tripped and "total" in (breaker.reason or "")


def test_portfolio_breaker_trips_on_daily_drawdown() -> None:
    breaker = PortfolioBreaker(100_000.0, max_total_drawdown=0.08, max_daily_drawdown=0.03)
    breaker.record("XAUUSD", -2_000.0)
    assert breaker.allowed()
    breaker.record("GBPUSD", -2_000.0)
    assert not breaker.allowed()  # 4% daily dd > 3%
    assert breaker.tripped and "daily" in (breaker.reason or "")
