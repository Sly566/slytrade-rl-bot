"""Tests for Layer 4 signal engine."""
from __future__ import annotations

from datetime import UTC, timedelta

import numpy as np
import pandas as pd
import pytest

from slytrade.strategy.config import ConfluenceConfig, ExitPlan, SessionFilter, SetupGrades, StrategyConfig
from slytrade.strategy.signals import (
    Signal,
    _evaluate_row,
    _grade,
    _killzone_tag,
    _runner_target,
    _strategy_columns,
)


def _permissive_cfg() -> StrategyConfig:
    """A permissive StrategyConfig that accepts all grades/zones/killzones (pre-battle-test defaults).
    Used by unit tests that validate engine mechanics independent of current trading persona."""
    return StrategyConfig(
        grades=SetupGrades(),
        exits=ExitPlan(
            tp1_r=1.0, tp1_pct=0.4, tp2_r=2.0, tp2_pct=0.3,
            ob_invalidation_buffer=0.1,
        ),
        sessions=SessionFilter(
            trade_asian_kz=True, trade_asian_range_retest=True,
        ),
        confluence=ConfluenceConfig(
            min_atr_pct=0.0002,
            min_risk_atr=0.0, max_risk_atr=100.0,
            accept_ob_tfs=("H1","M15","M5"),
            accept_zone_kinds=("OB","FVG"),
            accept_grades=("A+","A","B","C"),
            accept_longs=True, accept_shorts=True,
        ),
    )


UTC = UTC


def _empty_state():
    return {}


def _row(**kw):
    """Build a minimal row-dict behaving like a pd.Series for _evaluate_row."""
    base = {
        'time': pd.Timestamp('2025-01-10 13:30', tz='UTC'),
        'close': 3000.0, 'open': 3000.0, 'high': 3000.5, 'low': 2999.5,
        'atr_14': 2.0,
        'session': 'NY', 'kz_ny': True, 'kz_london': False, 'kz_asian': False,
        'london_open_30': False, 'ny_open_30': True,
        'vol_spike': False,
        # M5 trigger (neutral default — no displacement, no BOS/CHoCH)
        'M5_bull_disp': False, 'M5_bear_disp': False,
        'M5_minor_bos_up': False, 'M5_minor_bos_dn': False,
        'M5_major_bos_up': False, 'M5_major_bos_dn': False,
        'M5_major_choch_up': False, 'M5_major_choch_dn': False,
        'M5_major_bias': 0,
    }
    # Add all ob/fvg columns as NaN / mitigated=True so state sees no zones
    for tf in ('M5','M15','H1','H4','D1','W1','M30'):
        for side in ('bull','bear'):
            for kind in ('ob','fvg'):
                base[f'{tf}_{side}_{kind}_top'] = np.nan
                base[f'{tf}_{side}_{kind}_bottom'] = np.nan
                base[f'{tf}_{side}_{kind}_mitigated'] = True
            base[f'{tf}_major_bias'] = 0
            base[f'{tf}_major_swing_high'] = np.nan
            base[f'{tf}_major_swing_low'] = np.nan
            base[f'{tf}_price_in_range_pct'] = 0.5
    base.update(kw)
    return pd.Series(base)


class TestKillzone:
    def test_off_hours_blocked(self):
        cfg = StrategyConfig()
        row = _row(session='OFF', kz_ny=False, kz_london=False, kz_asian=False,
                   london_open_30=False, ny_open_30=False)
        allowed, tag = _killzone_tag(row, cfg.sessions)
        assert not allowed

    def test_ny_kz_allowed(self):
        cfg = StrategyConfig()
        row = _row(session='NY', kz_ny=True, ny_open_30=True)
        allowed, tag = _killzone_tag(row, cfg.sessions)
        assert allowed
        assert 'ny' in tag


class TestGrade:
    def test_a_plus_requires_D1_H4_H1_aligned(self):
        cfg = StrategyConfig()
        # Long, all required TFs bull
        row = _row(
            D1_major_bias=1, H4_major_bias=1, H1_major_bias=1,
            M15_major_bias=1, M5_major_bias=1,
            M15_price_in_range_pct=0.2,  # discount for long
        )
        grade, tags = _grade(1, row, cfg.confluence, bonus_killzone=False)
        assert grade == 'A+'
        assert 'M15_discount_long' in tags

    def test_c_grade(self):
        cfg = StrategyConfig()
        # Only M15 aligned (c_required_tfs)
        row = _row(M15_major_bias=1)
        grade, tags = _grade(1, row, cfg.confluence, bonus_killzone=False)
        assert grade == 'C'

    def test_fails_when_no_tf_agrees(self):
        cfg = StrategyConfig()
        row = _row()  # all biases 0
        grade, tags = _grade(1, row, cfg.confluence, bonus_killzone=False)
        assert grade == 'fail'


class TestEvaluateRow:
    def test_returns_none_without_trigger(self):
        cfg = StrategyConfig()
        row = _row()
        sig = _evaluate_row(0, row, cfg, _empty_state())
        assert sig is None

    def test_no_signal_without_zone(self):
        # Bull disp on M5 but no active OB/FVG -> no signal
        cfg = StrategyConfig()
        row = _row(M5_bull_disp=True, M5_major_bias=1,
                   H1_major_bias=1, H4_major_bias=1, D1_major_bias=1)
        sig = _evaluate_row(0, row, cfg, _empty_state())
        assert sig is None

    def test_fires_on_bull_ob_retest_in_killzone(self):
        cfg = _permissive_cfg()
        state = _empty_state()
        # First bar: M5 bull disp, H1 bull OB appears
        row0 = _row(M5_bull_disp=True, M5_major_bias=1,
                    H1_major_bias=1, H4_major_bias=1, D1_major_bias=1,
                    M15_major_bias=1, M15_price_in_range_pct=0.3,
                    H1_bull_ob_top=3010.0, H1_bull_ob_bottom=2995.0,
                    H1_bull_ob_mitigated=False,
                    close=3015.0)  # price above OB, no retest yet
        sig0 = _evaluate_row(0, row0, cfg, state)
        assert sig0 is None  # not in zone

        # Second bar: price retraces INTO the OB zone during NY kz
        row1 = _row(time=pd.Timestamp('2025-01-10 13:35', tz='UTC'),
                    M5_major_bias=1,
                    H1_major_bias=1, H4_major_bias=1, D1_major_bias=1,
                    M15_major_bias=1, M15_price_in_range_pct=0.3,
                    H1_bull_ob_top=3010.0, H1_bull_ob_bottom=2995.0,
                    H1_bull_ob_mitigated=False,
                    close=3005.0)  # inside OB
        sig1 = _evaluate_row(1, row1, cfg, state)
        assert sig1 is not None
        assert sig1.direction == 1
        assert sig1.stop < sig1.entry
        assert sig1.tp1 > sig1.entry
        assert sig1.tp2 > sig1.tp1
        assert sig1.risk_per_unit == pytest.approx(sig1.entry - sig1.stop, abs=1e-6)
        assert sig1.ob_tf == 'H1'
        assert sig1.grade in ('A+','A','B','C')
        # Stop must be below OB bottom (0.1 ATR buffer in permissive cfg, atr=2 → 0.2 below)
        assert sig1.stop < 2995.0
        # TP1 should be 1.0R above entry (permissive cfg)
        assert sig1.tp1 == pytest.approx(sig1.entry + sig1.risk_per_unit, rel=1e-3)

    def test_no_duplicate_on_same_zone(self):
        cfg = _permissive_cfg()
        state = _empty_state()
        for i, c in enumerate([3015.0, 3005.0, 3004.0, 3003.0]):
            row = _row(time=pd.Timestamp('2025-01-10 13:30', tz='UTC') + timedelta(minutes=5*i),
                       M5_bull_disp=(i==0),
                       M5_major_bias=1,
                       H1_major_bias=1, H4_major_bias=1, D1_major_bias=1,
                       M15_major_bias=1, M15_price_in_range_pct=0.3,
                       H1_bull_ob_top=3010.0, H1_bull_ob_bottom=2995.0,
                       H1_bull_ob_mitigated=False,
                       close=c)
            sig = _evaluate_row(i, row, cfg, state)
            if i == 1:
                assert sig is not None
            elif i > 1:
                assert sig is None  # same zone, no duplicate

    def test_bear_ob_retest(self):
        cfg = _permissive_cfg()
        state = _empty_state()
        # Bar 0: bear disp on M5, H1 bear OB appears below price
        row0 = _row(M5_bear_disp=True, M5_major_bias=-1,
                    H1_major_bias=-1, H4_major_bias=-1, D1_major_bias=-1,
                    M15_major_bias=-1, M15_price_in_range_pct=0.7,
                    H1_bear_ob_top=3005.0, H1_bear_ob_bottom=2990.0,
                    H1_bear_ob_mitigated=False,
                    close=2985.0)
        assert _evaluate_row(0, row0, cfg, state) is None
        # Bar 1: price rallies INTO the bear OB (retest)
        row1 = _row(time=pd.Timestamp('2025-01-10 13:35', tz='UTC'),
                    M5_major_bias=-1,
                    H1_major_bias=-1, H4_major_bias=-1, D1_major_bias=-1,
                    M15_major_bias=-1, M15_price_in_range_pct=0.7,
                    H1_bear_ob_top=3005.0, H1_bear_ob_bottom=2990.0,
                    H1_bear_ob_mitigated=False,
                    close=3000.0)
        sig = _evaluate_row(1, row1, cfg, state)
        assert sig is not None
        assert sig.direction == -1
        assert sig.stop > sig.entry > sig.tp1 > sig.tp2
        assert sig.stop > 3005.0  # buffer above OB top

    def test_bull_fvg_entry(self):
        cfg = _permissive_cfg()
        state = _empty_state()
        # bull FVG on M5: gap [bot=2998, top=3000]; bull disp on M5 at bar0
        row0 = _row(M5_bull_disp=True, M5_major_bias=1,
                    H1_major_bias=1, M15_major_bias=1,
                    M5_bull_fvg_top=3000.0, M5_bull_fvg_bottom=2998.0,
                    M5_bull_fvg_mitigated=False,
                    H1_bull_ob_top=np.nan, H1_bull_ob_bottom=np.nan,
                    close=3005.0)
        # Need H1 aligned for B-grade but C needs only M15
        assert _evaluate_row(0, row0, cfg, state) is None
        row1 = _row(time=pd.Timestamp('2025-01-10 13:35', tz='UTC'),
                    M5_major_bias=1,
                    H1_major_bias=1, M15_major_bias=1,
                    M5_bull_fvg_top=3000.0, M5_bull_fvg_bottom=2998.0,
                    M5_bull_fvg_mitigated=False,
                    H1_bull_ob_top=np.nan, H1_bull_ob_bottom=np.nan,
                    close=2999.0)
        sig = _evaluate_row(1, row1, cfg, state)
        assert sig is not None
        assert sig.fvg_top is not None and sig.ob_tf is None
        assert sig.direction == 1

    def test_emergency_choch_blocks_entry(self):
        cfg = _permissive_cfg()
        state = _empty_state()
        # Bull disp + H1 bull OB but M15 just had a bear CHoCH
        row0 = _row(M5_bull_disp=True, M5_major_bias=1,
                    H1_major_bias=1, M15_major_bias=-1,
                    H1_bull_ob_top=3010.0, H1_bull_ob_bottom=2995.0,
                    H1_bull_ob_mitigated=False,
                    M15_major_choch_dn=True,  # EMERGENCY CHoCH against long
                    close=3015.0)
        _evaluate_row(0, row0, cfg, state)
        row1 = _row(time=pd.Timestamp('2025-01-10 13:35', tz='UTC'),
                    M5_major_bias=1,
                    H1_major_bias=1, M15_major_bias=-1,
                    H1_bull_ob_top=3010.0, H1_bull_ob_bottom=2995.0,
                    H1_bull_ob_mitigated=False,
                    M15_major_choch_dn=True,
                    close=3005.0)
        sig = _evaluate_row(1, row1, cfg, state)
        assert sig is None

    def test_asian_kz_downgrades_to_c(self):
        cfg = _permissive_cfg()
        state = _empty_state()
        # Asia session, H4+H1 aligned (would be A) but should be downgraded to C
        row0 = _row(session='ASIA', kz_ny=False, kz_london=False, kz_asian=True,
                    london_open_30=False, ny_open_30=False,
                    M5_bull_disp=True, M5_major_bias=1,
                    H4_major_bias=1, H1_major_bias=1, D1_major_bias=1,
                    M15_major_bias=1, M15_price_in_range_pct=0.3,
                    M5_bull_ob_top=3002.0, M5_bull_ob_bottom=2998.0,
                    M5_bull_ob_mitigated=False,
                    close=3005.0)
        _evaluate_row(0, row0, cfg, state)
        row1 = _row(time=pd.Timestamp('2025-01-10 01:05', tz='UTC'),
                    session='ASIA', kz_ny=False, kz_london=False, kz_asian=True,
                    london_open_30=False, ny_open_30=False,
                    M5_major_bias=1,
                    H4_major_bias=1, H1_major_bias=1, D1_major_bias=1,
                    M15_major_bias=1, M15_price_in_range_pct=0.3,
                    M5_bull_ob_top=3002.0, M5_bull_ob_bottom=2998.0,
                    M5_bull_ob_mitigated=False,
                    close=3000.0)
        sig = _evaluate_row(1, row1, cfg, state)
        assert sig is not None
        assert sig.grade == 'C'
        assert 'asia_downgraded_to_c' in sig.confluence

    def test_duplicate_confluence_tags_not_emitted(self):
        """Regression: _grade must not add the same {TF}_bias_aligned twice."""
        cfg = _permissive_cfg()
        state = _empty_state()
        row0 = _row(M5_bull_disp=True, M5_major_bias=1,
                    H1_major_bias=1, M15_major_bias=1,
                    M5_bull_fvg_top=3002.0, M5_bull_fvg_bottom=2998.0,
                    M5_bull_fvg_mitigated=False,
                    close=3005.0)
        _evaluate_row(0, row0, cfg, state)
        row1 = _row(time=pd.Timestamp('2025-01-10 13:35', tz='UTC'),
                    M5_major_bias=1, H1_major_bias=1, M15_major_bias=1,
                    M5_bull_fvg_top=3002.0, M5_bull_fvg_bottom=2998.0,
                    M5_bull_fvg_mitigated=False,
                    close=3000.0)
        sig = _evaluate_row(1, row1, cfg, state)
        assert sig is not None
        # No tag appears more than once
        from collections import Counter
        dupes = [t for t,c in Counter(sig.confluence).items() if c > 1]
        assert dupes == [], f"duplicate tags: {dupes}"

    def test_signal_dataclass_r_multiples(self):
        s = Signal(time=pd.Timestamp('2025-01-10', tz='UTC'),
                   direction=1, entry=100.0, stop=99.0,
                   tp1=101.0, tp2=102.0, tp_runner=103.0,
                   risk_per_unit=1.0, grade='B', risk_pct=0.005)
        assert s.r_multiple_tp1 == pytest.approx(1.0)
        assert s.r_multiple_tp2 == pytest.approx(2.0)


class TestBOSControlAnchor:
    """BOS_CONT SL anchor must use the NEAREST opposing minor swing.

    v0.9.13.2 regression: the trigger-TF (M5) ATR-ZigZag level is ffilled and
    only refreshes when a new pivot confirms 1.5×ATR away. In a trend it can
    sit hours/35+ pts behind price (16:16 live: risk=39.38 vs atr=2.06 ≈ 19×
    ATR), so the old M5-first anchor always failed the 0.5–7 ATR band and
    BOS_CONT never fired.
    """

    def _cfg(self):
        return StrategyConfig(
            exits=ExitPlan(tp1_r=0.85, tp1_pct=0.4, tp2_r=2.0, tp2_pct=0.3,
                           ob_invalidation_buffer=0.1),
            sessions=SessionFilter(),  # NY KZ allowed (row is 13:30 NY)
            confluence=ConfluenceConfig(
                min_atr_pct=0.0001, min_risk_atr=0.0, max_risk_atr=100.0,
                accept_ob_tfs=("H1", "M15", "M5"),
                accept_zone_kinds=("OB", "FVG"),
                accept_grades=("A+", "A", "B", "C"),
                accept_longs=True, accept_shorts=True,
                persona_gating=False,
            ),
        )

    def test_uses_fresh_m1_anchor_when_m5_is_stale(self):
        cfg = self._cfg()
        row = _row(
            close=3000.0, atr_14=2.0,
            M5_minor_bos_up=True,
            M5_minor_swing_low=2985.0,   # hours old, 15 pts away (7.5× ATR)
            minor_swing_low=2998.0,      # the M1 level the bar broke FROM
        )
        sig = _evaluate_row(0, row, cfg, _empty_state())
        assert sig is not None, "stale M5 anchor must not kill BOS_CONT"
        assert sig.setup_kind == "BOS_CONT"
        assert sig.direction == 1
        # SL = fresh M1 swing - 0.1×ATR buffer; risk ≈ 2.2 (within 0.5-7 ATR)
        assert sig.stop == pytest.approx(2998.0 - 0.1 * 2.0)
        assert sig.risk_per_unit == pytest.approx(2.0 + 0.2)

    def test_keeps_trigger_tf_anchor_when_it_is_nearest(self):
        cfg = self._cfg()
        row = _row(
            close=3000.0, atr_14=2.0,
            M5_minor_bos_up=True,
            M5_minor_swing_low=2998.0,   # fresh M5 anchor, 2 pts away
            minor_swing_low=2997.0,      # M1 anchor is farther
        )
        sig = _evaluate_row(0, row, cfg, _empty_state())
        assert sig is not None
        assert sig.setup_kind == "BOS_CONT"
        assert sig.stop == pytest.approx(2998.0 - 0.1 * 2.0)

    def test_sl_clamp_rescues_stale_anchors(self):
        """v0.9.15: SL clamp [0.5 ATR, 3 ATR] brings far anchors within bounds."""
        cfg = self._cfg()
        row = _row(
            close=3000.0, atr_14=2.0,
            M5_minor_bos_up=True,
            M5_minor_swing_low=2985.0,   # 7.5× ATR away
            minor_swing_low=2986.0,      # also far
        )
        sig = _evaluate_row(0, row, cfg, _empty_state())
        # v0.9.15: SL clamp brings the stop to 3*2 = 6 pts from entry
        # so risk = 6 pts = 3 ATR, which is within the 0.5-7 ATR band
        assert sig is not None
        assert sig.setup_kind == "BOS_CONT"
        assert sig.risk_per_unit == pytest.approx(6.0, abs=0.1)  # clamped to 3 ATR

    def test_falls_back_to_m5_when_m1_anchor_missing(self):
        cfg = self._cfg()
        row = _row(
            close=3000.0, atr_14=2.0,
            M5_minor_bos_up=True,
            M5_minor_swing_low=2998.0,   # only M5 anchor exists
        )
        sig = _evaluate_row(0, row, cfg, _empty_state())
        assert sig is not None
        assert sig.setup_kind == "BOS_CONT"
        assert sig.stop == pytest.approx(2998.0 - 0.1 * 2.0)


class TestRunnerTarget:
    def test_picks_nearest_reasonable_swing(self):
        cfg = _permissive_cfg()
        row = pd.Series({
            'M15_major_bias': 1, 'M15_major_swing_high': 3008.0,
            'H1_major_bias': 1,  'H1_major_swing_high': 3020.0,
            'H4_major_bias': 1,  'H4_major_swing_high': 3050.0,
            'D1_major_bias': 1,  'D1_major_swing_high': 3200.0,
        })
        tf, px = _runner_target(1, row, cfg.confluence, entry=3000.0, risk=2.0)
        # M15 swing is 4R away, should be chosen as nearest
        assert tf == 'M15'
        assert px == pytest.approx(3008.0)

    def test_caps_distant_swing(self):
        cfg = _permissive_cfg()
        row = pd.Series({
            'M15_major_bias': 1, 'M15_major_swing_high': np.nan,  # no M15 target
            'H1_major_bias': 1,  'H1_major_swing_high': 3050.0,   # 25R away (too far)
            'H4_major_bias': 1,  'H4_major_swing_high': 3100.0,
            'D1_major_bias': 1,  'D1_major_swing_high': 3200.0,
        })
        tf, px = _runner_target(1, row, cfg.confluence, entry=3000.0, risk=2.0)
        # All swings > 8R -> 3R fallback; caller fills in 3R price
        assert tf == '3R_fallback'
        assert px is None  # caller computes 3R px when _runner_target returns None

    def test_ignores_swing_closer_than_tp2(self):
        """A swing closer than TP2 (2R) is a trap, not a runner target."""
        cfg = StrategyConfig()
        row = pd.Series({
            'M15_major_bias': 1, 'M15_major_swing_high': 3001.0,  # 0.5R away (<2R)
            'H1_major_bias': 1,  'H1_major_swing_high': 3010.0,   # 5R (good)
            'H4_major_bias': 1,  'H4_major_swing_high': 3050.0,
            'D1_major_bias': 1,  'D1_major_swing_high': 3200.0,
        })
        tf, px = _runner_target(1, row, cfg.confluence, entry=3000.0, risk=2.0)
        # Skip the too-close M15 swing; pick H1 at 5R
        assert tf == 'H1'
        assert px == pytest.approx(3010.0)


class TestStrategyColumns:
    def test_strategy_columns_has_trigger_tf_and_ob_tfs(self):
        cfg = StrategyConfig()
        cols = _strategy_columns(cfg)
        assert 'close' in cols
        assert 'atr_14' in cols
        assert 'M5_bull_disp' in cols
        assert 'H1_bull_ob_top' in cols
        assert 'D1_major_bias' in cols
        # sanity: fewer than all 535
        assert len(cols) < 300
