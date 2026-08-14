from __future__ import annotations

import pytest

from slytrade.risk.sizing import kelly_fraction, kelly_volume, normalize_volume, risk_based_volume


def test_normalize_volume_steps_and_clamps() -> None:
    assert normalize_volume(0.0) == 0.0
    assert normalize_volume(1.2345, volume_min=0.01, volume_step=0.01) == 1.23
    assert normalize_volume(9999.0, volume_max=100.0) == 100.0
    assert normalize_volume(0.001, volume_min=0.1, volume_step=0.1) == 0.1


def test_risk_based_volume_math() -> None:
    # Risk 0.5% of 100k = 500; stop distance 1.0 -> volume 500 (cap lifted).
    volume = risk_based_volume(100_000.0, 1.0, risk_per_trade=0.005, point_value=1.0, volume_max=10_000.0)
    assert volume == 500.0
    # Wider stop -> smaller size.
    assert risk_based_volume(100_000.0, 2.0, risk_per_trade=0.005, volume_max=10_000.0) == 250.0


def test_risk_based_volume_clamped_to_max() -> None:
    # Default volume_max (100 lots) caps an otherwise larger position.
    assert risk_based_volume(100_000.0, 1.0, risk_per_trade=0.005) == 100.0


def test_risk_based_volume_zero_on_bad_inputs() -> None:
    assert risk_based_volume(100_000.0, 0.0) == 0.0
    assert risk_based_volume(0.0, 1.0) == 0.0
    assert risk_based_volume(100_000.0, 1.0, risk_per_trade=0.0) == 0.0


def test_kelly_fraction_bounds() -> None:
    assert kelly_fraction(0.0, 1.0, 1.0) == 0.0
    assert kelly_fraction(1.0, 1.0, 1.0) == 0.0
    # p=0.6, b=2 -> f* = 0.6 - 0.4/2 = 0.4 (clamped to <= 0.5).
    assert kelly_fraction(0.6, 2.0, 1.0) == pytest.approx(0.4)
    assert kelly_fraction(0.5, 0.0, 1.0) == 0.0
    assert kelly_fraction(0.5, 1.0, 0.0) == 0.0


def test_kelly_volume_uses_fraction_of_kelly() -> None:
    volume = kelly_volume(
        100_000.0,
        1.0,
        win_rate=0.6,
        avg_win=2.0,
        avg_loss=1.0,
        kelly_fraction_of=0.25,
        volume_max=1_000_000.0,
    )
    # f* = 0.4; stake = 100k * 0.4 * 0.25 = 10k; stop 1.0 -> volume 10k.
    assert volume == pytest.approx(10_000.0)
