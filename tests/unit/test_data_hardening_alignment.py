import numpy as np
import pandas as pd

from slytrade.data.alignment import TICK_BAR_FEATURE_COLUMNS, align_market_data
from slytrade.features.ict import FEATURE_COLUMNS


def make_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-01", periods=4, freq="min", tz="UTC"),
            "symbol": "XAUUSDm",
            "timeframe": "M1",
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "tick_volume": [10, 10, 10, 10],
            "spread": [10, 10, 10, 10],
            "real_volume": [0, 0, 0, 0],
        }
    )


def make_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-01T00:00:00Z", periods=9, freq="30s").floor("s"),
            "time_msc": pd.date_range("2026-07-01T00:00:00Z", periods=9, freq="30s"),
            "symbol": "XAUUSD",
            "bid": np.linspace(100.0, 104.0, 9),
            "ask": np.linspace(100.2, 104.2, 9),
            "last": np.zeros(9),
            "volume": np.ones(9),
            "volume_real": np.zeros(9),
            "flags": np.zeros(9),
            "spread": np.full(9, 0.2),
            "mid": np.linspace(100.1, 104.1, 9),
        }
    )


def test_align_market_data_attaches_ict_and_tick_features():
    dataset = align_market_data(make_bars(), make_ticks(), timeframe="M1")

    for column in FEATURE_COLUMNS:
        assert column in dataset.bars.columns
    for column in TICK_BAR_FEATURE_COLUMNS:
        assert column in dataset.bars.columns
    assert dataset.bars["tick_count"].sum() > 0
    assert dataset.manifest.ict_feature_columns
    assert dataset.manifest.tick_feature_columns


def test_attach_tick_bar_features_uses_completed_bar_interval_only():
    dataset = align_market_data(make_bars(), make_ticks(), timeframe="M1", include_ict_features=False)
    bars = dataset.bars

    assert bars.loc[0, "tick_count"] == 3.0  # 00:00, 00:00:30, 00:01:00
    assert bars.loc[0, "tick_mid_close"] == 101.1


def test_fresh_coverage_ratio_manifest_warns_when_low():
    ticks = make_ticks().iloc[:1].copy()
    dataset = align_market_data(make_bars(), ticks, timeframe="M1", min_fresh_coverage=0.95)

    assert dataset.manifest.fresh_coverage_ratio >= 0.0
    assert dataset.manifest.quality_status == "WARN"
    assert dataset.manifest.quality_issues
