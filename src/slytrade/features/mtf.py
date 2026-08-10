"""Multi-timeframe ICT feature computation and causal timestamp alignment."""
import pandas as pd

from slytrade.features.ict import compute_ict_features


def compute_mtf_ict_features(m1_df: pd.DataFrame, higher_tf_data: dict) -> pd.DataFrame:
    if m1_df.empty:
        return m1_df.copy()
    if "time" not in m1_df.columns:
        raise ValueError("execution bars must contain a time column")

    result = m1_df.copy().sort_values("time")
    result["_mtf_row_time"] = pd.to_datetime(result["time"], utc=True)
    result["_mtf_original_index"] = result.index
    for tf, tf_df in higher_tf_data.items():
        if tf == "M1" or tf_df.empty:
            continue
        if "time" not in tf_df.columns:
            raise ValueError(f"{tf} bars must contain a time column")
        tf_features = compute_ict_features(tf_df.sort_values("time")).copy()
        prefix = f"htf_{tf.lower()}_"
        timeframe = str(tf).upper()
        durations = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}
        if timeframe not in durations:
            raise ValueError(f"unsupported higher timeframe: {tf}")
        tf_features["_mtf_feature_time"] = pd.to_datetime(tf_features["time"], utc=True) + pd.to_timedelta(
            durations[timeframe], unit="m"
        )
        feature_columns = [
            column for column in tf_features.columns if column not in {"time", "_mtf_feature_time", "symbol"}
        ]
        tf_features = tf_features[["_mtf_feature_time", *feature_columns]].rename(
            columns={column: f"{prefix}{column}" for column in feature_columns}
        )
        result = pd.merge_asof(
            result.sort_values("_mtf_row_time"),
            tf_features.sort_values("_mtf_feature_time"),
            left_on="_mtf_row_time",
            right_on="_mtf_feature_time",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns=["_mtf_feature_time"])
    result = result.sort_values("_mtf_original_index").drop(columns=["_mtf_row_time", "_mtf_original_index"])
    score = 0
    for tf in ["m5", "m15", "h1", "h4"]:
        col = f"htf_{tf}_bos_dir"
        if col in result.columns:
            score += (result[col] != 0).astype(int)
    result["mtf_confluence_score"] = score
    bias_cols = [c for c in result.columns if "htf_" in c and "bos_dir" in c]
    if bias_cols:
        result["mtf_bias"] = result[bias_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    else:
        result["mtf_bias"] = 0
    return result
