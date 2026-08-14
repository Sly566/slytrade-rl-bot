from __future__ import annotations

from datetime import UTC, datetime

from slytrade.runtime.circuit_breaker import LossCircuitBreaker, TradingLimits, limits_from_config


def _now(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def test_consecutive_losses_trigger_cooldown() -> None:
    breaker = LossCircuitBreaker(TradingLimits(max_consecutive_losses=3, cooldown_minutes=30), now=lambda: _now("2026-08-14T10:00:00+00:00"))
    assert breaker.check().allowed
    breaker.record_trade(-10.0)
    breaker.record_trade(-10.0)
    decision = breaker.record_trade(-10.0)
    assert not decision.allowed
    assert decision.reason == "cooldown after consecutive losses"
    assert breaker.paused


def test_cooldown_expires() -> None:
    clock = {"now": _now("2026-08-14T10:00:00+00:00")}
    breaker = LossCircuitBreaker(TradingLimits(max_consecutive_losses=2, cooldown_minutes=30), now=lambda: clock["now"])
    breaker.record_trade(-5.0)
    breaker.record_trade(-5.0)
    assert not breaker.check().allowed
    clock["now"] = _now("2026-08-14T10:31:00+00:00")
    assert breaker.check().allowed
    assert not breaker.paused


def test_win_resets_consecutive_streak() -> None:
    breaker = LossCircuitBreaker(TradingLimits(max_consecutive_losses=2), now=lambda: _now("2026-08-14T10:00:00+00:00"))
    breaker.record_trade(-5.0)
    breaker.record_trade(10.0)
    decision = breaker.record_trade(-5.0)
    assert decision.allowed


def test_daily_loss_cap() -> None:
    breaker = LossCircuitBreaker(TradingLimits(max_daily_losses=2), now=lambda: _now("2026-08-14T10:00:00+00:00"))
    breaker.record_trade(-5.0)
    breaker.record_trade(-5.0)
    assert not breaker.check().allowed
    assert breaker.check().reason == "max daily losses reached"


def test_daily_trade_cap() -> None:
    breaker = LossCircuitBreaker(TradingLimits(max_daily_trades=3), now=lambda: _now("2026-08-14T10:00:00+00:00"))
    breaker.record_trade(1.0)
    breaker.record_trade(1.0)
    breaker.record_trade(1.0)
    assert not breaker.check().allowed
    assert breaker.check().reason == "max daily trades reached"


def test_daily_counters_reset_next_day() -> None:
    clock = {"now": _now("2026-08-14T23:59:00+00:00")}
    breaker = LossCircuitBreaker(TradingLimits(max_daily_losses=1), now=lambda: clock["now"])
    breaker.record_trade(-5.0)
    assert not breaker.check().allowed
    clock["now"] = _now("2026-08-15T00:01:00+00:00")
    assert breaker.check().allowed


def test_limits_from_config_reads_risk_yaml_keys() -> None:
    limits = limits_from_config(
        {
            "max_consecutive_losses": 4,
            "max_daily_losses": 6,
            "max_daily_trades": 12,
            "cooldown_after_losses_minutes": 45,
            "risk_per_trade": 0.01,
            "max_daily_drawdown": 0.04,
            "max_total_drawdown": 0.10,
        }
    )
    assert limits.max_consecutive_losses == 4
    assert limits.cooldown_minutes == 45
    assert limits.risk_per_trade == 0.01


def test_cooldown_never_blocks_on_non_negative_pnl() -> None:
    breaker = LossCircuitBreaker(TradingLimits(max_consecutive_losses=1), now=lambda: _now("2026-08-14T10:00:00+00:00"))
    assert breaker.record_trade(0.0).allowed
    assert not breaker.paused
