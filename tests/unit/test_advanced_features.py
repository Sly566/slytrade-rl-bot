import numpy as np
import pandas as pd

from slytrade.ml.feature_selection import apply_feature_selection, select_features
from slytrade.ml.features import compute_ml_features
from slytrade.ml.volume_profile import compute_volume_profile_features
from slytrade.rl.ensemble import WeightedPolicyEnsemble


def make_bars(n: int = 80) -> pd.DataFrame:
    close = 100.0 + np.cumsum(np.sin(np.arange(n) / 5.0) * 0.2 + 0.05)
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": close,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "tick_volume": np.linspace(10.0, 100.0, n),
        }
    )


def test_volume_profile_and_ml_features_are_finite_and_causal():
    bars = make_bars()
    features = compute_ml_features(bars)
    assert {"vp_position", "vp_poc_distance_atr", "vp_volume_concentration"} <= set(features.columns)
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    prefix = compute_volume_profile_features(bars.iloc[:40])
    full = compute_volume_profile_features(bars)
    pd.testing.assert_frame_equal(prefix, full.iloc[:40], check_dtype=False)


def test_feature_selection_is_training_only_and_reusable():
    bars = make_bars()
    features = compute_ml_features(bars)
    selection = select_features(features, bars["close"], train_start=0, train_end=50, max_features=4)
    assert selection.train_end == 50
    assert len(selection.selected) == 4
    assert list(apply_feature_selection(features, selection).columns) == list(selection.selected)


class StubPolicy:
    def __init__(self, action: int):
        self.action = action

    def predict(self, observation: object) -> int:
        return self.action


def test_policy_ensemble_abstains_on_disagreement():
    ensemble = WeightedPolicyEnsemble((StubPolicy(1), StubPolicy(2)), min_confidence=0.75)
    decision = ensemble.predict_with_confidence(None)
    assert decision.abstained
    assert decision.action == 0
