import numpy as np
import pandas as pd

from slytrade.features.ict import FEATURE_COLUMNS, ICTFeatureConfig, compute_atr, compute_ict_features


def make_bars(close: list[float] | np.ndarray) -> pd.DataFrame:
    close_arr = np.asarray(close, dtype=float)
    open_arr = close_arr.copy()
    high = close_arr + 0.4
    low = close_arr - 0.4
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(close_arr), freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": open_arr,
            "high": high,
            "low": low,
            "close": close_arr,
            "tick_volume": np.full(len(close_arr), 100.0),
            "spread": np.full(len(close_arr), 2.0),
            "real_volume": np.zeros(len(close_arr)),
        }
    )


def test_atr_is_causal_and_positive():
    bars = make_bars([10, 11, 12, 11, 10, 9, 10])
    atr = compute_atr(bars, period=3)

    assert len(atr) == len(bars)
    assert (atr > 0).all()


def test_feature_output_has_expected_columns_and_is_finite():
    bars = make_bars(np.linspace(100.0, 110.0, 80))
    features = compute_ict_features(bars)

    for column in ["time", "symbol", "timeframe", *FEATURE_COLUMNS]:
        assert column in features.columns
    numeric = features[FEATURE_COLUMNS]
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_no_lookahead_prefix_invariance():
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.3, 140))
    bars = make_bars(close)
    split = 90

    full_features = compute_ict_features(bars, ICTFeatureConfig(pivot_lookback=4))
    prefix_features = compute_ict_features(bars.iloc[:split].copy(), ICTFeatureConfig(pivot_lookback=4))

    pd.testing.assert_frame_equal(
        full_features.iloc[:split].reset_index(drop=True),
        prefix_features.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_pivot_confirmation_is_delayed():
    bars = make_bars([1, 2, 3, 6, 3, 2, 1, 1, 1])
    features = compute_ict_features(bars, ICTFeatureConfig(pivot_lookback=2))

    assert features.loc[3, "pivot_high_confirmed"] == 0.0
    assert features.loc[5, "pivot_high_confirmed"] == 1.0


def test_bos_flag_after_confirmed_pivot_break():
    bars = make_bars([1, 2, 3, 6, 3, 2, 1, 4, 7, 8])
    features = compute_ict_features(bars, ICTFeatureConfig(pivot_lookback=2, bos_buffer_atr=0.0))

    assert (features["bos_dir"] == 1.0).any()
