"""Unit tests for v0.9.15 hybrid ladder + DISP_TRAP/BREAKER + SL clamp + limit retests.

Pins the new features introduced in v0.9.15:
  1. Hybrid ladder exits: TP1 1.0R @ 50% → BE; TP2 2.5R @ 25%; runner 25% ATR trail
  2. SL clamp: [0.5·ATR, min(3·ATR, 12pt)]
  3. DISP_TRAP setup detection
  4. BREAKER setup detection
  5. BOS_CONT forced to weak C grade
  6. Limit-at-zone for RETEST/BREAKER
  7. working_lot default 0.04
  8. risk_cap is only hard size rail (no 3× grade REJECT)
"""
from __future__ import annotations

from datetime import UTC, timedelta

import numpy as np
import pandas as pd
import pytest

from slytrade import __version__
from slytrade.strategy.config import (
    ConfluenceConfig,
    ExitPlan,
    SessionFilter,
    SetupGrades,
    StrategyConfig,
    champion_persona,
    rl_training_persona,
)
from slytrade.strategy.signals import (
    Signal,
    _clamp_sl,
    _evaluate_row,
    _grade,
)


# --------------------------------------------------------------------------- #
# Version
# --------------------------------------------------------------------------- #

class TestVersion:
    def test_version_is_0915(self):
        assert __version__ == "0.9.15"


# --------------------------------------------------------------------------- #
# ExitPlan defaults (hybrid ladder)
# --------------------------------------------------------------------------- #

class TestExitPlan:
    def test_tp1_is_1r_at_50pct(self):
        ep = ExitPlan()
        assert ep.tp1_r == 1.0
        assert ep.tp1_pct == 0.50

    def test_tp2_is_2r5_at_25pct(self):
        ep = ExitPlan()
        assert ep.tp2_r == 2.5
        assert ep.tp2_pct == 0.25

    def test_runner_trail_050_atr(self):
        ep = ExitPlan()
        assert ep.runner_trail_atr_mult == 0.5

    def test_sl_clamp_defaults(self):
        ep = ExitPlan()
        assert ep.sl_clamp_min_atr == 0.5
        assert ep.sl_clamp_max_atr == 2.5


# --------------------------------------------------------------------------- #
# SL clamp
# --------------------------------------------------------------------------- #

class TestSlClamp:
    def test_clamps_too_tight(self):
        """Stop 0.1 ATR away → clamped to 0.5 ATR."""
        ep = ExitPlan()
        result = _clamp_sl(3000.0, 2999.8, 1, 2.0, ep)  # 0.2 pts = 0.1 ATR
        assert abs(3000.0 - result) == pytest.approx(1.0)  # 0.5 * 2.0

    def test_clamps_too_wide(self):
        """Stop 5 ATR away → clamped to 2.5 ATR = 5 pts."""
        ep = ExitPlan()
        result = _clamp_sl(3000.0, 2990.0, 1, 2.0, ep)  # 10 pts = 5 ATR
        assert abs(3000.0 - result) == pytest.approx(5.0)  # 2.5*2

    def test_no_clamp_within_bounds(self):
        """Stop 2 ATR away → no clamping."""
        ep = ExitPlan()
        result = _clamp_sl(3000.0, 2996.0, 1, 2.0, ep)  # 4 pts = 2 ATR
        assert result == pytest.approx(2996.0)

    def test_clamp_short_direction(self):
        """Short: stop above entry, clamp symmetric."""
        ep = ExitPlan()
        result = _clamp_sl(3000.0, 3010.0, -1, 2.0, ep)  # 10 pts = 5 ATR
        assert abs(result - 3000.0) == pytest.approx(5.0)

    def test_no_absolute_cap_pts(self):
        """v0.9.15.1: removed absolute pts cap — purely ATR-based."""
        ep = ExitPlan()
        # ATR=100 (BTC-like), 2.5*ATR=250 — no artificial 12pt cap
        result = _clamp_sl(77000.0, 76700.0, 1, 100.0, ep)  # 300 pts = 3 ATR
        assert abs(77000.0 - result) == pytest.approx(250.0)


# --------------------------------------------------------------------------- #
# Working lot
# --------------------------------------------------------------------------- #

class TestWorkingLot:
    def test_default_working_lot(self):
        cfg = StrategyConfig()
        assert cfg.working_lot == 0.04

    def test_champion_working_lot(self):
        cfg = champion_persona()
        assert cfg.working_lot == 0.04


# --------------------------------------------------------------------------- #
# BOS_CONT forced to weak C
# --------------------------------------------------------------------------- #

class TestBosContGrade:
    def _cfg(self):
        return StrategyConfig(
            exits=ExitPlan(tp1_r=1.0, tp1_pct=0.5, tp2_r=2.5, tp2_pct=0.25),
            sessions=SessionFilter(),
            confluence=ConfluenceConfig(
                min_atr_pct=0.0001, min_risk_atr=0.0, max_risk_atr=100.0,
                accept_ob_tfs=("H1", "M15", "M5"),
                accept_zone_kinds=("OB", "FVG"),
                accept_grades=("A+", "A", "B", "C"),
                accept_longs=True, accept_shorts=True,
                persona_gating=False,
            ),
        )

    def test_bos_cont_forced_to_c(self):
        """BOS_CONT should always be C grade regardless of HTF alignment."""
        cfg = self._cfg()
        row = pd.Series({
            'time': pd.Timestamp('2025-01-10 13:30', tz='UTC'),
            'close': 3000.0, 'open': 3000.0, 'high': 3000.5, 'low': 2999.5,
            'atr_14': 2.0,
            'session': 'NY', 'kz_ny': True, 'kz_london': False, 'kz_asian': False,
            'london_open_30': False, 'ny_open_30': True,
            'vol_spike': True,
            'bull_disp': True, 'bear_disp': False,
            'minor_bos_up': True, 'minor_bos_dn': False,
            'minor_choch_up': False, 'minor_choch_dn': False,
            'major_bos_up': False, 'major_bos_dn': False,
            'major_choch_up': False, 'major_choch_dn': False,
            'bull_liq_sweep': False, 'bear_liq_sweep': False,
            'bull_sweep_px': np.nan, 'bear_sweep_px': np.nan,
            'minor_swing_high': np.nan, 'minor_swing_low': 2998.0,
            'M5_bull_disp': True, 'M5_bear_disp': False,
            'M5_minor_bos_up': True, 'M5_minor_bos_dn': False,
            'M5_minor_choch_up': False, 'M5_minor_choch_dn': False,
            'M5_major_bos_up': False, 'M5_major_bos_dn': False,
            'M5_major_choch_up': False, 'M5_major_choch_dn': False,
            'M5_major_bias': 1,
            'M5_minor_swing_low': 2998.0,
            'M5_vol_spike': True,
            'D1_major_bias': 1, 'H4_major_bias': 1, 'H1_major_bias': 1,
            'M15_major_bias': 1, 'M30_major_bias': 1,
        })
        # Add all ob/fvg columns as NaN / mitigated
        for tf in ('M5', 'M15', 'H1', 'H4', 'D1', 'W1', 'M30'):
            for side in ('bull', 'bear'):
                for kind in ('ob', 'fvg'):
                    row[f'{tf}_{side}_{kind}_top'] = np.nan
                    row[f'{tf}_{side}_{kind}_bottom'] = np.nan
                    row[f'{tf}_{side}_{kind}_mitigated'] = True
                row[f'{tf}_major_swing_high'] = np.nan
                row[f'{tf}_major_swing_low'] = np.nan
                row[f'{tf}_price_in_range_pct'] = 0.5
        state = {}
        sig = _evaluate_row(0, row, cfg, state)
        assert sig is not None
        assert sig.setup_kind == "BOS_CONT"
        assert sig.grade == 'C'  # always forced to C


# --------------------------------------------------------------------------- #
# Limit-at-zone for RETEST/BREAKER
# --------------------------------------------------------------------------- #

class TestLimitOrder:
    def test_retest_ob_uses_limit(self):
        """RETEST_OB signals should have use_limit_order=True."""
        cfg = StrategyConfig(
            exits=ExitPlan(tp1_r=1.0, tp1_pct=0.5, tp2_r=2.5, tp2_pct=0.25),
            sessions=SessionFilter(),
            confluence=ConfluenceConfig(
                min_atr_pct=0.0001, min_risk_atr=0.0, max_risk_atr=100.0,
                accept_ob_tfs=("H1", "M15", "M5"),
                accept_zone_kinds=("OB",),
                accept_grades=("A+", "A", "B", "C"),
                accept_longs=True, accept_shorts=False,
                persona_gating=False,
            ),
        )
        state = {}
        # Bar 0: M5 bull disp, H1 bull OB appears
        row0 = pd.Series({
            'time': pd.Timestamp('2025-01-10 13:30', tz='UTC'),
            'close': 3015.0, 'open': 3000.0, 'high': 3016.0, 'low': 2999.0,
            'atr_14': 2.0,
            'session': 'NY', 'kz_ny': True, 'kz_london': False, 'kz_asian': False,
            'london_open_30': False, 'ny_open_30': True,
            'vol_spike': False,
            'bull_disp': False, 'bear_disp': False,
            'minor_bos_up': False, 'minor_bos_dn': False,
            'minor_choch_up': False, 'minor_choch_dn': False,
            'major_bos_up': False, 'major_bos_dn': False,
            'major_choch_up': False, 'major_choch_dn': False,
            'bull_liq_sweep': False, 'bear_liq_sweep': False,
            'bull_sweep_px': np.nan, 'bear_sweep_px': np.nan,
            'minor_swing_high': np.nan, 'minor_swing_low': np.nan,
            'M5_bull_disp': True, 'M5_bear_disp': False,
            'M5_minor_bos_up': False, 'M5_minor_bos_dn': False,
            'M5_minor_choch_up': False, 'M5_minor_choch_dn': False,
            'M5_major_bos_up': False, 'M5_major_bos_dn': False,
            'M5_major_choch_up': False, 'M5_major_choch_dn': False,
            'M5_major_bias': 1,
            'M5_minor_swing_low': np.nan,
            'M5_vol_spike': False,
            'D1_major_bias': 1, 'H4_major_bias': 1, 'H1_major_bias': 1,
            'M15_major_bias': 1, 'M30_major_bias': 1,
            'H1_bull_ob_top': 3010.0, 'H1_bull_ob_bottom': 2995.0,
            'H1_bull_ob_mitigated': False,
        })
        for tf in ('M5', 'M15', 'H1', 'H4', 'D1', 'W1', 'M30'):
            for side in ('bull', 'bear'):
                for kind in ('ob', 'fvg'):
                    if f'{tf}_{side}_{kind}_top' not in row0:
                        row0[f'{tf}_{side}_{kind}_top'] = np.nan
                        row0[f'{tf}_{side}_{kind}_bottom'] = np.nan
                        row0[f'{tf}_{side}_{kind}_mitigated'] = True
                row0[f'{tf}_major_swing_high'] = np.nan
                row0[f'{tf}_major_swing_low'] = np.nan
                row0[f'{tf}_price_in_range_pct'] = 0.5
        _evaluate_row(0, row0, cfg, state)

        # Bar 1: price retraces INTO the OB zone
        row1 = row0.copy()
        row1['time'] = pd.Timestamp('2025-01-10 13:35', tz='UTC')
        row1['close'] = 3005.0
        row1['M5_bull_disp'] = False
        sig = _evaluate_row(1, row1, cfg, state)
        assert sig is not None
        assert sig.setup_kind == "RETEST_OB"
        assert sig.use_limit_order is True


# --------------------------------------------------------------------------- #
# DISP_TRAP setup
# --------------------------------------------------------------------------- #

class TestDispTrap:
    def test_disp_trap_not_blocked_by_persona(self):
        """DISP_TRAP should be accepted in the accept_setup_kinds."""
        cfg = champion_persona()
        assert "DISP_TRAP" in cfg.confluence.accept_setup_kinds

    def test_breaker_not_blocked_by_persona(self):
        """BREAKER should be accepted in the accept_setup_kinds."""
        cfg = champion_persona()
        assert "BREAKER" in cfg.confluence.accept_setup_kinds


# --------------------------------------------------------------------------- #
# Hybrid ladder LiveTrade fields
# --------------------------------------------------------------------------- #

class TestLiveTradeHybridLadder:
    def test_live_trade_has_hybrid_fields(self):
        from slytrade.live.trader import LiveTrade
        from datetime import datetime
        lt = LiveTrade(
            ticket=1, direction=1, entry=3000.0, sl=2995.0, tp=3005.0,
            lots=0.04, open_time=datetime.now(UTC), grade='B', risk_pct=0.005,
            tp2_price=3012.5, remaining_lots=0.04, original_lots=0.04,
        )
        assert lt.tp1_hit is False
        assert lt.tp2_hit is False
        assert lt.tp2_price == 3012.5
        assert lt.remaining_lots == 0.04
        assert lt.original_lots == 0.04
        assert lt.runner_trail_px == 0.0


# --------------------------------------------------------------------------- #
# risk_cap only hard rail (no 3× REJECT)
# --------------------------------------------------------------------------- #

class TestVolMinRiskV0915:
    def test_no_3x_reject(self):
        """v0.9.15: risk_cap is the only hard rail, no 3× target REJECT."""
        from slytrade.live.trader import LiveTrader
        from slytrade.backtest.specs import AccountSpec, spec_for_symbol
        from types import SimpleNamespace

        class FakeMT5:
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            ORDER_FILLING_IOC = 1
            ORDER_FILLING_RETURN = 2
            TRADE_ACTION_DEAL = 1
            ORDER_TIME_GTC = 0
            TRADE_RETCODE_DONE = 10009
            TRADE_RETCODE_DONE_PARTIAL = 10010
            def account_info(self): return SimpleNamespace(equity=3000.0, balance=3000.0, currency="ZAR", leverage=2000)
            def symbol_info_tick(self, _): return SimpleNamespace(bid=4600.0, ask=4600.3)
            def positions_get(self, **kw): return []

        spec = spec_for_symbol("XAUUSDm", overrides={
            "point": 0.001, "digits": 3, "contract_size": 100.0,
            "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01,
            "currency_profit": "USD", "tick_value": 0.10,
        })
        acct = AccountSpec(starting_equity=3000.0, currency="ZAR", leverage=2000,
                          fx_to_account={"USD": 18.5})
        trader = LiveTrader(
            mt5=FakeMT5(), symbol="XAUUSDm", spec=spec,
            cfg=champion_persona(), acct=acct,
            live=False, risk_cap=0.02, max_open=3,
        )
        # 1.2% actual vs 0.3% target = 4× — under risk_cap=2% → OK (no 3× REJECT)
        assert trader._vol_min_risk_ok(0.012, 0.003, silent=True) is True
        # 2.1% actual > risk_cap=2% but < 3x risk_cap=6% → ACCEPT (v0.9.15.15 min-lot safety)
        assert trader._vol_min_risk_ok(0.021, 0.003, silent=True) is True
        # 7% actual > 3x risk_cap=6% → REJECT
        assert trader._vol_min_risk_ok(0.07, 0.003, silent=True) is False
