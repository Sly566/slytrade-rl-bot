"""Unit tests for v0.9.13 / v0.9.13.1 / v0.9.13.2 live-trader risk + orphan-adoption helpers.

These pin the incomplete-merge regressions that bit the first v0.9.13 land:
  1. vol_min floor must HARD-REJECT when actual risk > max(risk_cap, 1.5%) or 3× target
  2. orphan adoption must seed bars_held from wall-clock age (time-stop continuity)
  3. broker open-time parsing must accept unix ints and datetime objects
  4. _clamp_sleep must guarantee time.sleep NEVER gets a negative duration
     (the 13:14 ValueError crash in the bar-boundary poll loop)
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from slytrade.backtest.specs import AccountSpec, spec_for_symbol
from slytrade.live.trader import MAGIC, LiveTrade, LiveTrader
from slytrade.strategy.config import champion_persona

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

class _FakeMT5:
    """Minimal stub — no network. Only methods the helpers touch."""

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010

    def __init__(self, positions: list[dict] | None = None, equity: float = 3000.0,
                 history_deals: list[dict] | None = None):
        self._positions = list(positions or [])
        self._equity = equity
        self._history = list(history_deals or [])

    def history_deals_select(self, *_args, **_kwargs) -> bool:
        return True

    def history_deals_get(self, **kwargs: Any) -> list[SimpleNamespace]:
        pos_id = kwargs.get("position", -1)
        out = []
        for d in self._history:
            if int(d.get("position", -1)) == int(pos_id):
                out.append(SimpleNamespace(**d))
        return out

    def account_info(self) -> SimpleNamespace:
        return SimpleNamespace(equity=self._equity, balance=self._equity, currency="ZAR", leverage=2000)

    def symbol_info_tick(self, _symbol: str) -> SimpleNamespace:
        return SimpleNamespace(bid=4600.0, ask=4600.3)

    def positions_get(self, **kwargs: Any) -> list[SimpleNamespace]:
        out = []
        for p in self._positions:
            if "ticket" in kwargs and int(p.get("ticket", -1)) != int(kwargs["ticket"]):
                continue
            if "symbol" in kwargs and p.get("symbol") and p["symbol"] != kwargs["symbol"]:
                continue
            out.append(SimpleNamespace(**p))
        return out

    def order_send(self, _req: dict) -> SimpleNamespace:
        return SimpleNamespace(retcode=10009, order=1, deal=1, comment="ok")

    def shutdown(self) -> None:
        pass


@pytest.fixture()
def trader() -> LiveTrader:
    mt5 = _FakeMT5(equity=3000.0)
    spec = spec_for_symbol(
        "XAUUSDm",
        overrides={
            "point": 0.001,
            "digits": 3,
            "contract_size": 100.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "currency_profit": "USD",
            "tick_value": 0.10,  # $0.10 per point per lot
        },
    )
    acct = AccountSpec(
        starting_equity=3000.0,
        currency="ZAR",
        leverage=2000,
        fx_to_account={"USD": 18.5},
    )
    return LiveTrader(
        mt5=mt5,
        symbol="XAUUSDm",
        spec=spec,
        cfg=champion_persona(),
        acct=acct,
        live=False,
        risk_cap=0.01,
        max_open=3,
        verbose=False,
    )


# --------------------------------------------------------------------------- #
# vol_min hard-REJECT thresholds
# --------------------------------------------------------------------------- #

class TestVolMinRiskOk:
    def test_rejects_when_actual_above_1_5pct_cap(self, trader: LiveTrader):
        # 3% actual vs 0.15% target → must REJECT (the 12:28 short case)
        assert trader._vol_min_risk_ok(0.03, 0.0015, silent=True) is False

    def test_rejects_when_actual_above_3x_target_even_if_under_cap(self, trader: LiveTrader):
        # risk_cap=1%, actual=1.2% is under the 1.5% floor-cap BUT 1.2/0.3%=4× target
        # → still REJECT via the 3× rule
        assert trader._vol_min_risk_ok(0.012, 0.003, silent=True) is False

    def test_allows_mild_1_25x_oversize(self, trader: LiveTrader):
        # 0.4% actual vs 0.3% target = 1.33× — under 1.5% cap and under 3× → OK (SIZE-WARN path)
        assert trader._vol_min_risk_ok(0.004, 0.003, silent=True) is True

    def test_allows_exact_target(self, trader: LiveTrader):
        assert trader._vol_min_risk_ok(0.005, 0.005, silent=True) is True

    def test_rejects_at_exactly_3x_plus_epsilon(self, trader: LiveTrader):
        assert trader._vol_min_risk_ok(0.0091, 0.003, silent=True) is False  # 3.03×

    def test_allows_just_under_3x_and_under_cap(self, trader: LiveTrader):
        # 0.0089 / 0.003 ≈ 2.97× and < 1.5% → OK
        assert trader._vol_min_risk_ok(0.0089, 0.003, silent=True) is True

    def test_respects_higher_risk_cap(self, trader: LiveTrader):
        # With risk_cap raised to 2%, max_risk_cap = 2%; 1.8% actual vs 1% target
        # is under 2% and under 3× → OK
        trader.risk_cap = 0.02
        assert trader._vol_min_risk_ok(0.018, 0.01, silent=True) is True
        # 2.1% still REJECTS against the 2% cap
        assert trader._vol_min_risk_ok(0.021, 0.01, silent=True) is False


# --------------------------------------------------------------------------- #
# Orphan open-time parsing + bars_held seeding
# --------------------------------------------------------------------------- #

class TestOrphanAge:
    def test_parse_unix_int(self):
        ts = int(datetime(2026, 8, 27, 12, 27, tzinfo=UTC).timestamp())
        got = LiveTrader._parse_broker_time(ts)
        assert got == datetime(2026, 8, 27, 12, 27, tzinfo=UTC)

    def test_parse_aware_datetime(self):
        dt = datetime(2026, 8, 27, 12, 27, tzinfo=UTC)
        assert LiveTrader._parse_broker_time(dt) == dt

    def test_parse_naive_datetime_assumes_utc(self):
        dt = datetime(2026, 8, 27, 12, 27)  # naive
        got = LiveTrader._parse_broker_time(dt)
        assert got.tzinfo is not None
        assert got.replace(tzinfo=None) == dt

    def test_bars_held_from_3h_age(self):
        now = datetime(2026, 8, 27, 15, 27, tzinfo=UTC)
        open_t = now - timedelta(hours=3)
        assert LiveTrader._orphan_bars_held(open_t, now=now) == 180  # 3h * 60

    def test_bars_held_fresh_is_zero(self):
        now = datetime(2026, 8, 27, 15, 27, tzinfo=UTC)
        assert LiveTrader._orphan_bars_held(now, now=now) == 0

    def test_bars_held_never_negative(self):
        now = datetime(2026, 8, 27, 15, 0, tzinfo=UTC)
        future = now + timedelta(minutes=5)
        assert LiveTrader._orphan_bars_held(future, now=now) == 0

    def test_adopt_seeds_bars_held_and_books_trade(self, trader: LiveTrader):
        open_t = datetime.now(UTC) - timedelta(hours=2, minutes=15)  # 135 min
        pos = {
            "ticket": 424242,
            "type": 0,  # BUY
            "price_open": 4600.0,
            "sl": 4595.0,
            "tp": 4604.0,
            "volume": 0.01,
            "time": int(open_t.timestamp()),
            "magic": MAGIC,
            "symbol": "XAUUSDm",
            "profit": 12.0,
        }
        lt = trader._adopt_orphan_position(424242, pos)
        assert lt is not None
        assert isinstance(lt, LiveTrade)
        assert lt.ticket == 424242
        assert lt.direction == 1
        assert lt.lots == 0.01
        assert lt.grade == "?"
        assert 130 <= lt.bars_held <= 140  # ~135 min, allow a few seconds of skew
        assert 424242 in trader._trades
        # Second adopt is a no-op (idempotent)
        assert trader._adopt_orphan_position(424242, pos) is None

    def test_adopt_short_direction(self, trader: LiveTrader):
        pos = {
            "ticket": 7,
            "type": 1,  # SELL
            "price_open": 4600.0,
            "sl": 4605.0,
            "tp": 4595.0,
            "volume": 0.02,
            "time": int(datetime.now(UTC).timestamp()),
            "magic": MAGIC,
            "symbol": "XAUUSDm",
            "profit": -3.0,
        }
        lt = trader._adopt_orphan_position(7, pos)
        assert lt is not None
        assert lt.direction == -1
        assert lt.lots == 0.02


# --------------------------------------------------------------------------- #
# Realized P&L from broker deal history (was last-poll estimate -> 4x understated)
# --------------------------------------------------------------------------- #

class TestDealProfit:
    def test_returns_realized_from_history(self, trader: LiveTrader):
        mt5 = _FakeMT5(
            history_deals=[
                {"position": 3148230553, "profit": -37.75, "commission": -0.0, "swap": 0.0},
            ]
        )
        trader.mt5 = mt5
        # The 16:14 live SL in v0.9.13.1: bot recorded -9.60ZAR (last-poll
        # unrealized) but the broker really lost -37.75ZAR.
        assert trader._deal_profit(3148230553) == pytest.approx(-37.75)

    def test_returns_none_when_no_deals(self, trader: LiveTrader):
        assert trader._deal_profit(999) is None

    def test_exit_prefers_deal_history_over_last_poll_estimate(self, trader: LiveTrader):
        """Regression: a stop filled between polls must book the REALIZED
        broker loss, not the stale unrealized profit from the previous poll."""
        ticket = 3148230553
        trader.live = True
        mt5 = _FakeMT5(
            positions=[],  # position already closed broker-side
            history_deals=[
                {"position": ticket, "profit": -37.75, "commission": 0.0, "swap": 0.0},
            ],
        )
        trader.mt5 = mt5
        lt = LiveTrade(
            ticket=ticket, direction=1, entry=4606.919, sl=4604.424, tp=4609.040,
            lots=0.01, open_time=datetime.now(UTC), grade="A", risk_pct=0.0131,
        )
        trader._trades[ticket] = lt
        # Previous poll saw the position still open at an unrealized -9.60.
        trader._last_broker_pos = {ticket: {"profit": -9.60, "magic": MAGIC,
                                            "symbol": "XAUUSDm", "ticket": ticket}}

        trader._monitor_positions(None)

        assert lt.closed is True
        assert lt.close_reason == "BROKER"
        assert lt.pnl == pytest.approx(-37.75)  # ground truth, NOT -9.60

    def test_exit_falls_back_to_last_poll_when_no_history(self, trader: LiveTrader):
        ticket = 7
        trader.live = True
        trader.mt5 = _FakeMT5(positions=[])  # no history deals
        lt = LiveTrade(
            ticket=ticket, direction=-1, entry=4600.0, sl=4605.0, tp=4595.0,
            lots=0.01, open_time=datetime.now(UTC), grade="C", risk_pct=0.005,
        )
        trader._trades[ticket] = lt
        trader._last_broker_pos = {ticket: {"profit": -3.0, "magic": MAGIC,
                                            "symbol": "XAUUSDm", "ticket": ticket}}
        trader._monitor_positions(None)
        assert lt.pnl == pytest.approx(-3.0)


# --------------------------------------------------------------------------- #
# _clamp_sleep: time.sleep must NEVER get a negative duration (13:14 crash)
# --------------------------------------------------------------------------- #

class TestClampSleep:
    def test_negative_remaining_clamps_to_zero(self):
        # The 13:14 crash: clock crossed end_wait mid-iteration, so
        # end_wait - time.time() went negative and time.sleep raised ValueError.
        assert LiveTrader._clamp_sleep(-0.5) == 0.0
        assert LiveTrader._clamp_sleep(-1e-9) == 0.0

    def test_negative_remaining_with_cap_clamps_to_zero(self):
        # Old code: min(poll_interval, negative) → negative still reaches sleep
        assert LiveTrader._clamp_sleep(-0.5, max_seconds=5.0) == 0.0

    def test_positive_duration_unchanged(self):
        assert LiveTrader._clamp_sleep(4.2) == 4.2

    def test_cap_applies(self):
        assert LiveTrader._clamp_sleep(7.0, max_seconds=5.0) == 5.0
        # cap >= duration → unchanged
        assert LiveTrader._clamp_sleep(4.2, max_seconds=5.0) == 4.2

    def test_zero_is_fine(self):
        assert LiveTrader._clamp_sleep(0.0) == 0.0
        assert LiveTrader._clamp_sleep(0.0, max_seconds=5.0) == 0.0

    def test_nan_and_inf_collapse_to_zero(self):
        assert LiveTrader._clamp_sleep(float("nan")) == 0.0
        assert LiveTrader._clamp_sleep(float("inf")) == 0.0
        assert LiveTrader._clamp_sleep(float("-inf")) == 0.0

    def test_garbage_collapses_to_zero(self):
        assert LiveTrader._clamp_sleep(None) == 0.0
        assert LiveTrader._clamp_sleep("not-a-number") == 0.0

    def test_negative_or_nan_cap_clamps_safe(self):
        assert LiveTrader._clamp_sleep(5.0, max_seconds=-1.0) == 0.0
        assert LiveTrader._clamp_sleep(5.0, max_seconds=float("nan")) == 0.0
        # +inf cap == no cap
        assert LiveTrader._clamp_sleep(5.0, max_seconds=float("inf")) == 5.0


# --------------------------------------------------------------------------- #
# End-to-end: sizing math that triggered the original 12:28 incident
# --------------------------------------------------------------------------- #

class TestSizingIncidentRegression:
    def test_wide_stop_minlot_would_have_been_3pct(self, trader: LiveTrader):
        """Reproduce the 12:28 short: 5pt stop, 0.01 min lot, ~R3000 equity, C-grade 0.15%.

        profit_per_lot(5.0) = 5/0.001 * 0.10 = $500 per lot
        0.01 lot → $5 risk → ×18.5 ZAR = R92.5 ≈ 3.08% of R3000.
        Target was 0.15% → must REJECT.
        """
        equity = 3000.0
        risk_per_unit = 5.0  # points of price
        target_risk_pct = 0.0015
        risk_acct = target_risk_pct * equity
        risk_quote = risk_acct / trader.acct.fx_to_account["USD"]
        lots = trader.spec.lots_for_risk(risk_per_unit, risk_quote)
        # vol_min floors to 0.01
        assert lots == pytest.approx(0.01)
        actual_risk_quote = trader.spec.profit_per_lot(risk_per_unit) * lots
        actual_risk_acct = trader.acct.to_account_ccy(actual_risk_quote, trader.spec.currency_profit)
        actual_risk_pct = actual_risk_acct / equity
        assert actual_risk_pct == pytest.approx(0.030833, rel=1e-3)
        assert trader._vol_min_risk_ok(actual_risk_pct, target_risk_pct, silent=True) is False
