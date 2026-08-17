"""Tests for the per-timeframe adaptive strategy profiles."""
from __future__ import annotations

from slytrade.config.timeframe_profiles import DEFAULT_PROFILE, profile_for
from slytrade.tasks import _persona_config_from_risk, _trade_config_from_risk


def test_profile_for_known_timeframes() -> None:
    assert profile_for("H1").min_score == 4
    assert profile_for("H1").cooldown_bars == 20
    assert profile_for("M15").min_score == 3
    assert profile_for("M15").take_profit_atr == 3.0
    assert profile_for("M15").max_bars_in_trade == 60
    # M5/M1 exist but are flagged unprofitable.
    assert profile_for("M5").min_score == 3
    assert profile_for("M1").min_score == 4


def test_profile_for_unknown_falls_back_to_champion() -> None:
    assert profile_for("ZZ9") is DEFAULT_PROFILE
    assert profile_for(None) is DEFAULT_PROFILE


def test_config_loaders_apply_profiles() -> None:
    h1_tc = _trade_config_from_risk("H1")
    assert h1_tc.stop_loss_atr == 1.0
    assert h1_tc.take_profit_atr == 2.0
    assert h1_tc.max_bars_in_trade is None

    m15_tc = _trade_config_from_risk("M15")
    assert m15_tc.take_profit_atr == 3.0
    assert m15_tc.max_bars_in_trade == 60

    h1_pc = _persona_config_from_risk("XAUUSD", "H1")
    m15_pc = _persona_config_from_risk("XAUUSD", "M15")
    assert h1_pc.min_score == 4 and h1_pc.cooldown_bars == 20
    assert m15_pc.min_score == 3 and m15_pc.cooldown_bars == 10
    # Timeframe-insensitive gates still come from risk.yaml.
    assert h1_pc.strict_mtf_direction is True
    assert m15_pc.require_entry_momentum is True
    # The H4 trend gate comes from the profile (validated default).
    assert h1_pc.htf_trend_timeframe == "h4"
    assert m15_pc.htf_trend_timeframe == "h4"


def test_profiles_use_symbol_point_value() -> None:
    pc = _persona_config_from_risk("XAUUSD", "H1")
    assert pc.point_value == 100.0  # gold
    pc_fx = _persona_config_from_risk("EURUSD", "H1")
    assert pc_fx.point_value == 1.0  # FX
