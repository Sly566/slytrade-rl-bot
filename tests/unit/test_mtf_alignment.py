import numpy as np
import pandas as pd

from slytrade.features.mtf import compute_mtf_ict_features


def _bars(periods: int, timeframe: str, freq: str) -> pd.DataFrame:
    close = np.linspace(100.0, 110.0, periods)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=periods, freq=freq, tz="UTC"),
            "symbol": "XAUUSD",
            "timeframe": timeframe,
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "tick_volume": 100.0,
        }
    )


def test_mtf_alignment_is_timestamp_based_and_preserves_rows():
    execution = _bars(20, "M1", "min")
    higher = _bars(4, "M5", "5min")

    result = compute_mtf_ict_features(execution, {"M5": higher})

    assert len(result) == len(execution)
    assert result.index.tolist() == execution.index.tolist()
    assert "htf_m5_bos_dir" in result.columns
    # The first five one-minute bars occur before the first M5 bar closes.
    assert result.loc[:4, "htf_m5_bos_dir"].isna().all()


def test_mtf_prefix_does_not_use_future_higher_timeframe_bars():
    execution = _bars(20, "M1", "min")
    higher = _bars(4, "M5", "5min")
    full = compute_mtf_ict_features(execution, {"M5": higher})
    prefix = compute_mtf_ict_features(execution.iloc[:12], {"M5": higher})

    pd.testing.assert_frame_equal(
        full.iloc[:12].reset_index(drop=True),
        prefix.reset_index(drop=True),
        check_dtype=False,
    )
