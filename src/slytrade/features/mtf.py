"""Multi-timeframe ICT feature computation and merging (causal)."""
import pandas as pd

from slytrade.features.ict import compute_ict_features


def compute_mtf_ict_features(m1_df: pd.DataFrame, higher_tf_data: dict) -> pd.DataFrame:
    result = m1_df.copy()
    for tf, tf_df in higher_tf_data.items():
        if tf == "M1" or tf_df.empty:
            continue
        tf_features = compute_ict_features(tf_df)
        prefix = f"htf_{tf.lower()}_"
        tf_features = tf_features.add_prefix(prefix)
        result = result.join(tf_features, how="left")
        result = result.ffill()
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
