"""Build scoped RL datasets and environment factories.

The central no-leakage rule: any normalization (e.g. the feature scaler) is
fitted only on the training slice of the data. ``RLDataset`` holds the raw
features; each walk-forward fold fits its own scaler on its train window.

The RL "superbrain" consumes the FULL validated feature set produced by the
pipeline — the ML features PLUS the ICT/SMC features, per-bar tick
microstructure, session flags and the multi-timeframe (htf_*/mtf_bias/
mtf_confluence_score) columns already embedded in the aligned bars. Nothing is
thrown away and nothing unseen is invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.data.alignment import TICK_BAR_FEATURE_COLUMNS
from slytrade.features.ict import FEATURE_COLUMNS as ICT_FEATURE_COLUMNS
from slytrade.intelligence.market_context import MarketContextEngine
from slytrade.intelligence.regime import MarketRegimeEngine
from slytrade.ml.features import ML_FEATURE_COLUMNS, compute_ml_features
from slytrade.rl.environment import RLEnvironmentConfig, SlyTradeRLEnvironment
from slytrade.rl.mode_vector import build_mode_vector

# Columns that are part of the aligned bars but are NOT trading features (they
# are either keys, prices, or already projected into the feature set).
_NON_FEATURE_PREFIXES = (
    "quote_",
)
_NON_FEATURE_COLUMNS = {
    "time",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "decision_time",
    "tick_volume",
    "spread",
    "real_volume",
    "quote_is_fresh",
}


def rl_feature_columns(bars: pd.DataFrame) -> list[str]:
    """Return the ordered feature columns the RL agent should observe.

    ML features are always present. On top of them, every validated feature
    column the aligned bars already carry is adopted: ICT/SMC, per-bar tick
    microstructure, sessions, and the multi-timeframe (htf_*, mtf_bias,
    mtf_confluence_score) columns.
    """
    columns: list[str] = []
    seen: set[str] = set()

    for column in ML_FEATURE_COLUMNS:
        if column in bars.columns and column not in seen:
            columns.append(column)
            seen.add(column)

    for group in (
        ICT_FEATURE_COLUMNS,
        TICK_BAR_FEATURE_COLUMNS,
        ("mtf_bias", "mtf_confluence_score"),
    ):
        for column in group:
            if column in bars.columns and column not in seen:
                columns.append(column)
                seen.add(column)

    for column in bars.columns:
        if column in seen or column in _NON_FEATURE_COLUMNS:
            continue
        if column.startswith(_NON_FEATURE_PREFIXES):
            continue
        if column.startswith(("htf_", "session_")):
            columns.append(column)
            seen.add(column)

    return columns


@dataclass(frozen=True)
class RLDataset:
    """Raw feature + bar matrices for RL training (no fitted scaler yet).

    Each fold fits its own scaler on its own train window to prevent leakage.
    """

    bars: pd.DataFrame
    features: pd.DataFrame
    symbol: str
    point_size: float
    point_value: float

    @property
    def feature_columns(self) -> list[str]:
        return list(self.features.columns)

    def fit_scaler(
        self,
        train_start: int,
        train_end: int,
        *,
        variance_floor: float = 1e-6,
    ) -> dict[str, tuple[float, float]]:
        """Fit per-column (mean, std) on features[train_start:train_end] only.

        Computed column-by-column on the slice view (never a full-frame copy),
        so a full-dataset scaler fit stays O(one column) in memory.
        """
        if not 0 <= train_start < train_end <= len(self.features):
            raise ValueError(f"invalid train slice [{train_start}, {train_end}) for {len(self.features)} rows")
        rng = np.random.default_rng(0)
        n_rows = train_end - train_start
        params: dict[str, tuple[float, float]] = {}
        for col in self.features.columns:
            series = pd.to_numeric(self.features[col].iloc[train_start:train_end], errors="coerce").fillna(0.0)
            std = float(series.std())
            if std < variance_floor:
                series = series + rng.normal(0.0, variance_floor, size=n_rows)
                std = float(series.std())
            params[col] = (float(series.mean()), std if std > 1e-9 else 1.0)
        return params

    def env_factory(
        self,
        start: int,
        end: int,
        *,
        seed: int,
        scaler_params: dict[str, tuple[float, float]],
        config: RLEnvironmentConfig | None = None,
        mode_vector: np.ndarray | None = None,
        feature_columns: list[str] | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> SlyTradeRLEnvironment:
        """Create an environment over bars[start:end] with the given scaler.

        ``feature_columns`` restricts the observation to a dynamic selection
        (fitted on the training slice only); when omitted the full feature set
        is used. ``candidate_mask`` (0/1/2 per bar) enables the "RL as a
        filter" mode: the env only accepts new entries on candidate bars.
        """
        if not 0 <= start < end <= len(self.bars):
            raise ValueError(f"invalid slice [{start}, {end}) for {len(self.bars)} rows")
        cfg = config or RLEnvironmentConfig(point_size=self.point_size, point_value=self.point_value, seed=seed)
        bars_slice = self.bars.iloc[start:end]
        if feature_columns is not None:
            missing = set(feature_columns).difference(self.features.columns)
            if missing:
                raise ValueError(f"selected features missing: {sorted(missing)}")
            columns = [column for column in self.features.columns if column in feature_columns]
        else:
            columns = list(self.features.columns)

        # Scale into a fresh float32 matrix (one allocation, no float64
        # temporaries, no per-column pandas copies). The env indexes rows
        # positionally, so no index reset is needed.
        matrix = self.features.iloc[start:end].loc[:, columns].to_numpy(dtype=np.float32)
        for column_index, column in enumerate(columns):
            mean, std = scaler_params[column]
            matrix[:, column_index] = (matrix[:, column_index] - float(mean)) / float(std)
        features_slice = pd.DataFrame(matrix, columns=columns)
        return SlyTradeRLEnvironment(
            features=features_slice,
            bars=bars_slice,
            config=cfg,
            mode_vector=mode_vector,
            candidate_mask=(candidate_mask[start:end] if candidate_mask is not None else None),
        )

    def select_features_on_fold(
        self,
        train_start: int,
        train_end: int,
        *,
        correlation_threshold: float = 0.92,
    ) -> tuple[str, ...]:
        """Dynamically select features on the training slice only.

        The objectives are the market-footprint outcomes (structure R-multiple
        and sweep reversal) from :mod:`slytrade.ml.footprint`. The count emerges
        from significance vs. shuffled shadows — no admin-set number.
        """
        from slytrade.ml.feature_selection import select_features_dynamic
        from slytrade.ml.footprint import structure_r_objective, sweep_reversal_objective

        if not 0 <= train_start < train_end <= len(self.bars):
            raise ValueError(f"invalid train slice [{train_start}, {train_end}) for {len(self.bars)} rows")
        bars_slice = self.bars.iloc[train_start:train_end]
        objectives = {
            "structure_r": structure_r_objective(bars_slice),
            "sweep_reversal": sweep_reversal_objective(bars_slice),
        }
        selection = select_features_dynamic(
            self.features,
            objectives,
            train_start=train_start,
            train_end=train_end,
            correlation_threshold=correlation_threshold,
        )
        return selection.selected


def infer_timeframe(bars: pd.DataFrame, default: str = "H1") -> str:
    """Return the decision timeframe of an aligned bars frame (for profiles)."""
    if "timeframe" in bars.columns and len(bars) > 0:
        value = bars["timeframe"].iloc[0]
        if value:
            return str(value).upper()
    return default


def _num(bars: pd.DataFrame, column: str) -> pd.Series:
    if column in bars.columns:
        return pd.to_numeric(bars[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=bars.index)


def persona_signal_columns(bars: pd.DataFrame) -> pd.DataFrame:
    """The persona-adaptive strategy's confluence score, pre-computed per bar.

    Mirrors ``SlyTradeRLEnvironment._setup_score`` / ``ICTConfluenceStrategy``
    exactly (BOS ±2, CHOCH ±1, sweep ±1, FVG ±1, order block ±1, premium/
    discount ±1, trend ±1), causally. ``persona_score`` is max(long, short);
    ``persona_bias`` is sign(long − short) — the direction a pro would take.
    """
    bos = _num(bars, "bos_dir").to_numpy(dtype=float)
    choch = _num(bars, "choch_dir").to_numpy(dtype=float)
    sweep = _num(bars, "liquidity_sweep").to_numpy(dtype=float)
    fvg_bull = _num(bars, "fvg_bullish").to_numpy(dtype=float)
    fvg_bear = _num(bars, "fvg_bearish").to_numpy(dtype=float)
    ob_bull = _num(bars, "order_block_bullish").to_numpy(dtype=float)
    ob_bear = _num(bars, "order_block_bearish").to_numpy(dtype=float)
    pd_ = _num(bars, "premium_discount").to_numpy(dtype=float)
    trend = _num(bars, "trend_strength").to_numpy(dtype=float)

    long_score = (
        (bos > 0) * 2.0
        + (choch > 0)
        + (sweep < 0)
        + (fvg_bull > 0)
        + (ob_bull > 0)
        + (pd_ <= -0.15)
        + (trend > 0)
    )
    short_score = (
        (bos < 0) * 2.0
        + (choch < 0)
        + (sweep > 0)
        + (fvg_bear > 0)
        + (ob_bear > 0)
        + (pd_ >= 0.15)
        + (trend < 0)
    )
    return pd.DataFrame(
        {
            "persona_score": np.maximum(long_score, short_score).astype(np.float32),
            "persona_bias": np.sign(long_score - short_score).astype(np.float32),
        },
        index=bars.index,
    )


def persona_action_column(bars: pd.DataFrame) -> pd.Series:
    """The persona-adaptive champion's ACTUAL entries, mapped to bars.

    0 = hold, 1 = enter long, 2 = enter short. Generated by running the managed
    backtest (the champion's own engine with its cooldown + side state + gates),
    so it reproduces the champion's decisions faithfully. A naive direct
    ``on_bar`` loop would never reset the persona's side after a managed exit
    and would under-generate entries by ~10x. This is the target the RL must
    learn to filter.
    """
    from slytrade.rl.walkforward import persona_actions_for_bars

    actions = np.asarray(persona_actions_for_bars(bars), dtype=np.float32)
    return pd.Series(actions, index=bars.index, name="persona_action")


def build_rl_dataset(bars: pd.DataFrame, personality: TraderPersonality | None = None) -> RLDataset:
    """Build a raw (unscaled) dataset from validated bars sorted by time.

    The bars must be the pipeline's aligned output: canonical OHLCV plus the
    validated feature columns. ML features are computed on top; the ICT/tick/
    MTF/session features already present in the bars are adopted verbatim; and
    the persona + market-regime mode matrix is appended so the policy is
    conditioned on *who it is* and *what the market is doing*.
    """
    if bars.empty:
        raise ValueError("bars frame is empty")
    required = {"time", "symbol", "open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    personality = personality or TraderPersonality.from_yaml()
    # The aligned output is already time-sorted with a RangeIndex; only pay for
    # a sort+copy when it genuinely isn't (a full-frame copy is ~1 GB).
    time_col = bars["time"]
    if not (time_col.is_monotonic_increasing and isinstance(bars.index, pd.RangeIndex)):
        bars = bars.sort_values("time").reset_index(drop=True)
    ml_features = compute_ml_features(bars)
    if ml_features.empty or len(ml_features) != len(bars):
        raise ValueError("feature computation failed")

    feature_columns = rl_feature_columns(bars)
    adopted = [column for column in feature_columns if column in bars.columns and column not in ml_features.columns]

    # Persona + regime conditioning channel (the \"professional trader\" wiring).
    from slytrade.rl.mode_vector import build_mode_matrix

    mode = build_mode_matrix(bars, personality).reset_index(drop=True)

    # Build the feature frame in one shot with concat: per-column assignment
    # fragments the DataFrame and emits a PerformanceWarning for every column
    # (the aligned bars carry ~200 adopted columns). Single-shot concat keeps
    # the frame contiguous and the console clean. Everything is float32 from
    # the start so the whole RL stack stays at half the float64 footprint.
    frames = [ml_features.reset_index(drop=True).astype(np.float32)]
    if adopted:
        adopted_frame = bars[adopted].reset_index(drop=True)
        adopted_frame = adopted_frame.apply(pd.to_numeric, errors="coerce").astype(np.float32)
        frames.append(adopted_frame)
    # Persona confluence signal: the SAME score + direction the rule-based
    # champion trades from, pre-computed causally per bar. Giving the RL this
    # composed signal (instead of expecting a small MLP to re-derive it from
    # 200+ raw columns) turns its job from "rediscover the edge" into "learn
    # when to filter the edge". ``persona_action`` additionally carries the
    # champion's stateful decision (cooldown/side-aware), which is what makes
    # behavioural cloning actually reproduce the champion.
    frames.append(persona_signal_columns(bars))
    frames.append(persona_action_column(bars).to_frame().astype(np.float32))
    frames.append(mode.astype(np.float32))
    features = pd.concat(frames, axis=1)
    features = features.fillna(0.0)

    return RLDataset(
        bars=bars,
        features=features,
        symbol=str(bars["symbol"].iloc[0]),
        point_size=0.01,
        point_value=1.0,
    )


def build_mode_vector_from_bars(
    personality: TraderPersonality,
    bars: pd.DataFrame,
    *,
    index: int,
) -> np.ndarray:
    """Build the mode vector for a specific bar index using history until it.

    Safe for training-time feature generation; the context engine consumes only
    bars up to `index`.
    """
    window = bars.iloc[max(0, index - 120) : index + 1]
    if window.empty:
        window = bars.iloc[0:1]
    context_engine = MarketContextEngine(personality, MarketRegimeEngine())
    context = context_engine.analyze(window)
    return build_mode_vector(personality, context)
