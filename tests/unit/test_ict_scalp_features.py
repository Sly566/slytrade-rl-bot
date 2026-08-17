"""Tests for the ICT/SMC scalping features (displacement, VI, IFVG, breaker,
draw-on-liquidity levels, kill zones)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.features.ict import ICTFeatureConfig, compute_ict_features

NEW_COLS = [
    "displacement_dir", "displacement_strength",
    "vi_bullish", "vi_bearish",
    "ifvg_bullish", "ifvg_bearish",
    "breaker_bullish", "breaker_bearish",
    "nearest_pdh_dist_atr", "nearest_pdl_dist_atr",
    "pdh_tap", "pdl_tap",
    "session_high_dist_atr", "session_low_dist_atr",
    "killzone_london", "killzone_ny", "killzone_london_close",
]


def make_bars(close: np.ndarray, hours: list[int] | None = None, start: str = "2026-01-05 00:00") -> pd.DataFrame:
    n = len(close)
    open_ = close.copy()
    high = close + 0.4
    low = close - 0.4
    times = pd.date_range(start, periods=n, freq="min", tz="UTC")
    if hours is not None:
        times = pd.to_datetime([f"2026-01-05 {h:02d}:{i % 60:02d}" for i, h in enumerate(hours)], utc=True)
    return pd.DataFrame(
        {
            "time": times,
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": np.full(n, 100.0),
            "spread": np.full(n, 2.0),
            "real_volume": np.zeros(n),
        }
    )


def test_new_columns_present_and_finite():
    bars = make_bars(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, 200)))
    f = compute_ict_features(bars)
    for c in NEW_COLS:
        assert c in f.columns, c
    assert np.isfinite(f[NEW_COLS].to_numpy(dtype=float)).all()


def test_displacement_flags_impulsive_candle():
    # Flat 100.0 for many bars, then a huge one-bar jump up.
    close = np.full(30, 100.0)
    close[29] = 102.0
    bars = make_bars(close)
    bars.loc[29, "open"] = 100.0
    bars.loc[29, "high"] = 102.05
    bars.loc[29, "low"] = 99.95
    f = compute_ict_features(bars)
    assert f.loc[29, "displacement_dir"] == 1.0
    assert f.loc[29, "displacement_strength"] > 1.0
    assert f.loc[28, "displacement_dir"] == 0.0


def test_vi_flags_body_gap():
    # bodies of candle i and i-2 don't overlap with a gap -> volume imbalance.
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    open_ = np.array([99.9, 99.9, 99.9, 99.9, 99.9, 99.9])
    # candle 2: body gap above candle 0 (body_bot[2]=100 > body_top[0]=99.9)
    # make candle 2 open=100.5 close=101.0 -> body_bot=100.5 > 99.9
    open_[2] = 100.5
    close[2] = 101.0
    bars = pd.DataFrame({
        "time": pd.date_range("2026-01-05", periods=6, freq="min", tz="UTC"),
        "symbol": "XAUUSD", "timeframe": "M1",
        "open": open_, "high": close + 0.2, "low": close - 0.2, "close": close,
        "tick_volume": np.full(6, 100.0), "spread": 2.0, "real_volume": 0.0,
    })
    f = compute_ict_features(bars, ICTFeatureConfig(vi_min_atr=0.01))
    assert f.loc[2, "vi_bullish"] == 1.0


def test_killzone_flags_by_hour():
    hours = [7, 9, 12, 14, 15, 16, 20]
    close = np.full(len(hours), 100.0)
    bars = make_bars(close, hours=hours)
    f = compute_ict_features(bars)
    assert f.loc[0, "killzone_london"] == 1.0 and f.loc[1, "killzone_london"] == 1.0
    assert f.loc[2, "killzone_ny"] == 1.0 and f.loc[3, "killzone_ny"] == 1.0
    assert f.loc[4, "killzone_london_close"] == 1.0 and f.loc[5, "killzone_london_close"] == 1.0
    assert f.loc[6, "killzone_london"] == 0.0 and f.loc[6, "killzone_ny"] == 0.0


def test_pdh_tap_after_previous_day_high():
    # Two days; second day price sits just under day-1's high.
    close = np.array([100.0, 101.0, 100.5, 100.8, 100.9, 100.95, 100.97, 101.0])
    times = pd.to_datetime(["2026-01-05 23:57", "2026-01-05 23:58", "2026-01-05 23:59",
                             "2026-01-06 00:00", "2026-01-06 00:01", "2026-01-06 00:02",
                             "2026-01-06 00:03", "2026-01-06 00:04"], utc=True)
    bars = pd.DataFrame({
        "time": times, "symbol": "XAUUSD", "timeframe": "M1",
        "open": close, "high": close + 0.05, "low": close - 0.05, "close": close,
        "tick_volume": np.full(8, 100.0), "spread": 2.0, "real_volume": 0.0,
    })
    f = compute_ict_features(bars, ICTFeatureConfig(tap_zone_atr=0.5))
    # Day-1 high = 101.0 (+0.05 wick); on 2026-01-06 price ~101.0 is within tap.
    assert f.loc[7, "nearest_pdh_dist_atr"] >= 0.0
    # PDH distance is small on the second day (price near 101.0)
    assert abs(f.loc[7, "nearest_pdh_dist_atr"]) < 1.5
    # First day has no previous day -> zeroed.
    assert f.loc[0, "nearest_pdh_dist_atr"] == 0.0


def test_no_lookahead_for_new_columns():
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.3, 140))
    bars = make_bars(close)
    split = 90
    full = compute_ict_features(bars, ICTFeatureConfig(pivot_lookback=4))
    prefix = compute_ict_features(bars.iloc[:split].copy(), ICTFeatureConfig(pivot_lookback=4))
    pd.testing.assert_frame_equal(
        full.iloc[:split].reset_index(drop=True),
        prefix.reset_index(drop=True),
        check_exact=False, rtol=1e-6, atol=1e-6,
    )
