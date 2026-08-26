"""Layer 5 backtest engine unit tests — synthetic data only."""
from __future__ import annotations

import pandas as pd
import pytest

from slytrade.backtest.positions import Direction, ExitReason, Position, Tranche
from slytrade.backtest.specs import SymbolSpec, AccountSpec, spec_for_symbol
from slytrade.backtest.engine import BacktestConfig, BacktestEngine


# ---------------------------------------------------------------------------
# Symbol / account specs (XAUUSDm per Exness demo)
# ---------------------------------------------------------------------------
@pytest.fixture()
def spec() -> SymbolSpec:
    return spec_for_symbol("XAUUSDm")


@pytest.fixture()
def acct() -> AccountSpec:
    return AccountSpec(starting_equity=100_000.0, currency="USD", fx_to_account={"USD": 1.0})


@pytest.fixture()
def bt_cfg() -> BacktestConfig:
    # Generous equity so every test signal gets opened without min-lot floor effects
    return BacktestConfig(
        starting_equity=100_000.0, account_ccy="USD",
        slippage_points_long=0, slippage_points_short=0,
        commission_per_lot_rt=0.0, max_open_positions=5, max_risk_per_trade=0.05,
    )


def _make_engine(spec, acct, bt_cfg) -> BacktestEngine:
    return BacktestEngine(spec, acct, bt_cfg=bt_cfg)


# ---------------------------------------------------------------------------
# Tranche / position P&L directionality (regression for the "shorts counted
# backwards" bug that gave 98% short win rate)
# ---------------------------------------------------------------------------
def test_long_profit_when_price_above_entry(spec):
    t = Tranche(name="T1", size_frac=1.0, lots=1.0, entry=2500.0, sl=2490.0, tp=2510.0)
    t.exit_price = 2510.0
    t.exit_reason = ExitReason.TP1
    t.state = "closed"
    # $10 move * 100 oz * 1 lot = $1000 profit
    assert t.realized_pnl_ccy(Direction.LONG, spec) == pytest.approx(1000.0, abs=0.01)


def test_long_loss_when_price_below_entry(spec):
    t = Tranche(name="T1", size_frac=1.0, lots=1.0, entry=2500.0, sl=2490.0, tp=2510.0)
    t.exit_price = 2490.0
    t.exit_reason = ExitReason.SL
    t.state = "closed"
    assert t.realized_pnl_ccy(Direction.LONG, spec) == pytest.approx(-1000.0, abs=0.01)


def test_short_profit_when_price_below_entry(spec):
    t = Tranche(name="T1", size_frac=1.0, lots=1.0, entry=2500.0, sl=2510.0, tp=2490.0)
    t.exit_price = 2490.0
    t.exit_reason = ExitReason.TP1
    t.state = "closed"
    assert t.realized_pnl_ccy(Direction.SHORT, spec) == pytest.approx(1000.0, abs=0.01)


def test_short_loss_when_price_above_entry(spec):
    t = Tranche(name="T1", size_frac=1.0, lots=1.0, entry=2500.0, sl=2510.0, tp=2490.0)
    t.exit_price = 2510.0
    t.exit_reason = ExitReason.SL
    t.state = "closed"
    assert t.realized_pnl_ccy(Direction.SHORT, spec) == pytest.approx(-1000.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tranche allocation respects broker volume granularity (regression for the
# 0-lot ghost tranche bug that left only T3 with real size)
# ---------------------------------------------------------------------------
def test_tranche_allocation_full_ladder(spec):
    p = Position(
        pos_id=1, symbol="XAUUSDm", direction=Direction.LONG,
        entry_time=pd.Timestamp("2024-01-01", tz="UTC"), entry_price=2500.0,
        total_lots=1.0, atr_at_entry=2.0, grade="A", risk_pct=0.01,
        risk_per_unit_quote=10.0, initial_sl=2490.0, tp1=2510.0, tp2=2520.0,
        tp_runner=2540.0, swing_target_tf="H1", swing_target_price=2540.0,
        trigger_tf="M5", ob_tf="M5", zone_kind="OB", killzone="ny_kz", session="NY",
    )
    p.init_tranches(volume_min=0.01, volume_step=0.01, t1_frac=0.5, t2_frac=0.3, t3_frac=0.2)
    lots = {t.name: t.lots for t in p.tranches}
    assert abs(sum(lots.values()) - 1.0) < 1e-6
    # All three tranches must have positive size when total_lots >= 3*volume_min
    assert lots["T1"] > 0 and lots["T2"] > 0 and lots["T3"] > 0


def test_tranche_allocation_micro_single(spec):
    """0.01 lots (single micro-lot) must collapse to a single T1 tranche."""
    p = Position(
        pos_id=2, symbol="XAUUSDm", direction=Direction.LONG,
        entry_time=pd.Timestamp("2024-01-01", tz="UTC"), entry_price=2500.0,
        total_lots=0.01, atr_at_entry=2.0, grade="C", risk_pct=0.0025,
        risk_per_unit_quote=5.0, initial_sl=2495.0, tp1=2505.0, tp2=2510.0,
        tp_runner=2520.0, swing_target_tf="", swing_target_price=2520.0,
        trigger_tf="M5", ob_tf=None, zone_kind="FVG", killzone="asian_kz_c_only",
        session="ASIA",
    )
    p.init_tranches(volume_min=0.01, volume_step=0.01)
    lots = {t.name: t.lots for t in p.tranches}
    assert sum(lots.values()) == pytest.approx(0.01, abs=1e-6)
    # No tranche may have 0 lots (would be a ghost)
    for t in p.tranches:
        assert t.lots > 0, f"tranche {t.name} allocated 0 lots"


# ---------------------------------------------------------------------------
# End-to-end engine: 1 long that hits TP1 on the next bar → close_reason set
# ---------------------------------------------------------------------------
def _bars_long_win() -> pd.DataFrame:
    """Three M1 bars: signal fires on bar 0; bar 1 high hits TP1."""
    return pd.DataFrame([
        {"time": pd.Timestamp("2024-01-02 13:30", tz="UTC"),
         "open": 2500.0, "high": 2500.2, "low": 2499.8, "close": 2500.0,
         "spread": 0, "tick_volume": 100, "atr_14": 2.0, "M5_atr_14": 2.5,
         "M5_major_choch_up": False, "M5_major_choch_dn": False,
         "M15_major_choch_up": False, "M15_major_choch_dn": False},
        {"time": pd.Timestamp("2024-01-02 13:31", tz="UTC"),
         "open": 2500.0, "high": 2510.5, "low": 2499.5, "close": 2510.0,
         "spread": 0, "tick_volume": 100, "atr_14": 2.0, "M5_atr_14": 2.5,
         "M5_major_choch_up": False, "M5_major_choch_dn": False,
         "M15_major_choch_up": False, "M15_major_choch_dn": False},
        {"time": pd.Timestamp("2024-01-02 13:32", tz="UTC"),
         "open": 2510.0, "high": 2510.5, "low": 2509.5, "close": 2510.0,
         "spread": 0, "tick_volume": 100, "atr_14": 2.0, "M5_atr_14": 2.5,
         "M5_major_choch_up": False, "M5_major_choch_dn": False,
         "M15_major_choch_up": False, "M15_major_choch_dn": False},
    ])


def _signal_long_tp1() -> pd.DataFrame:
    return pd.DataFrame([{
        "time": pd.Timestamp("2024-01-02 13:30", tz="UTC"),
        "direction": 1, "entry": 2500.0, "stop": 2490.0,
        "tp1": 2510.0, "tp2": 2520.0, "tp_runner": 2540.0,
        "risk_per_unit": 10.0, "grade": "A", "risk_pct": 0.01,
        "confluence": ["trigger_M5", "zone_OB_on_M5"],
        "fails": [], "trigger_tf": "M5",
        "ob_tf": "M5", "ob_top": 2500.5, "ob_bottom": 2499.5,
        "fvg_top": None, "fvg_bottom": None,
        "swing_target_tf": "H1", "swing_target_price": 2540.0,
        "atr_at_entry": 2.0, "htf_bias_summary": {"H1": 1, "M15": 1},
        "session": "NY", "killzone": "ny_kz",
    }])


def test_engine_long_hits_tp1_sets_close_reason(spec, acct, bt_cfg):
    engine = _make_engine(spec, acct, bt_cfg)
    # Give enough warmup that we don't need 500 bars — run directly via _process_bar
    engine._warmup_done = True  # type: ignore[attr-defined]
    bars = _bars_long_win()
    sigs = _signal_long_tp1()
    # Manually feed: first bar fires signal
    from collections import defaultdict
    sig_by_time: dict = defaultdict(list)
    for rec in sigs.itertuples(index=False):
        sig_by_time[rec.time].append(rec)

    for i in range(501):
        # burn warmup by calling _process_bar with no signal on a dummy row,
        # but simpler: short-circuit warmup via setting the internal flag through
        # the warmup counter by replacing the run loop
        pass

    # Easier: use engine.run() by writing the bars to a temp parquet and
    # reading back. Skip the warmup by inserting 500 filler rows.
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        warm = pd.DataFrame([{
            "time": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=i),
            "open": 2500.0, "high": 2500.1, "low": 2499.9, "close": 2500.0,
            "spread": 0, "tick_volume": 100, "atr_14": 2.0, "M5_atr_14": 2.5,
            "M5_major_choch_up": False, "M5_major_choch_dn": False,
            "M15_major_choch_up": False, "M15_major_choch_dn": False,
        } for i in range(600)])
        # Offset the signal bars so they fall AFTER warmup (501st bar onwards)
        t0 = warm["time"].iloc[-1] + pd.Timedelta(minutes=1)
        bars = _bars_long_win().copy()
        bars["time"] = [t0 + pd.Timedelta(minutes=i) for i in range(len(bars))]
        sig = _signal_long_tp1().copy()
        sig["time"] = [bars["time"].iloc[0]]
        full = pd.concat([warm, bars], ignore_index=True)
        fpath = pathlib.Path(td) / "part-0.parquet"
        full.to_parquet(fpath, index=False)
        engine._pos_id = 0
        engine.positions = []
        engine.closed = []
        engine.equity = acct.starting_equity
        engine.balance = acct.starting_equity
        engine.margin_used_quote = 0.0
        engine._eq_rows = []
        res = engine.run([fpath], sig)

    assert len(res.trades) == 1, f"expected 1 trade, got {len(res.trades)}"
    tr = res.trades.iloc[0]
    assert tr["direction"] == "LONG"
    # Close reason must NOT be NaN (regression for the close_reason bug)
    assert tr["close_reason"] in {"tp1", "tp2", "trail", "runner_target", "sl", "be", "time_stop"}, \
        f"close_reason missing/NaN: {tr['close_reason']!r}"
    # Trade should be a winner (hit TP1)
    assert tr["pnl_acct"] > 0, f"expected winning long, got pnl={tr['pnl_acct']}"
    # zone_kind must be OB (regression for NaN truthiness bug)
    assert tr["zone_kind"] == "OB"


# ---------------------------------------------------------------------------
# zone_kind helper
# ---------------------------------------------------------------------------
def test_zone_kind_distinguishes_ob_from_fvg(spec, acct, bt_cfg):
    class _Sig:
        def __init__(self, ob): self.ob_tf = ob
    assert BacktestEngine._zone_kind(_Sig("M5")) == "OB"
    assert BacktestEngine._zone_kind(_Sig(None)) == "FVG"
    import numpy as np
    assert BacktestEngine._zone_kind(_Sig(np.nan)) == "FVG"
