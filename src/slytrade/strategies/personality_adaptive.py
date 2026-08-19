"""Personality-adaptive ICT strategy.

The engine of the "ICT trader persona": the same causal ICT confluence scores
from the baseline strategy, but entry thresholds, direction filters and
position sizing are modulated in real time by the trader's personality traits
and the detected market regime.

Everything here is causal: decisions at bar *i* only use data from bars
`<= i`. The strategy keeps an internal rolling window so it can run inside any
of the existing backtest engines that call `on_bar(index, bar)`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from slytrade.config.trader_personality import TraderPersonality
from slytrade.execution.models import OrderIntent, OrderKind, Side
from slytrade.intelligence.market_context import MarketContextEngine
from slytrade.intelligence.micro_macro_alignment import MicroMacroAlignmentEngine
from slytrade.intelligence.regime import MarketRegimeEngine
from slytrade.strategies.baselines import ICTConfluenceStrategy

ADAPTIVE_RULE_KEYS = {
    "high": ["high_volatility", "high_volatility_selectivity"],
    "low": ["low_volatility"],
    "london": ["london_open"],
    "ny": ["ny_open"],
}


@dataclass(frozen=True)
class PersonalityAdaptiveConfig:
    """Configuration for the personality-adaptive strategy."""

    min_score: int = 4
    cooldown_bars: int = 20
    allowed_sessions: tuple[str, ...] = ("london", "ny_am", "ny_pm")
    require_fresh_quote: bool = True
    max_spread: float | None = None
    min_tick_rate: float = 0.0
    min_abs_trend_strength: float = 0.0
    history_window: int = 120
    # Risk-based position sizing
    risk_based_sizing: bool = True
    equity: float = 100_000.0
    risk_per_trade: float = 0.005
    point_value: float = 1.0
    stop_loss_atr: float = 1.0
    min_slot_distance_price: float = 0.10
    # Limit-entry: >0 rests a limit order at ``limit_entry_atr`` * ATR below the
    # close (longs) / above it (shorts) instead of market-entering. Measured on
    # 25mo real XAUUSD M15: a 0.25*ATR pullback limit fills ~97% of setups and
    # lifts net edge ~+72% (gold retraces 0.25ATR after a setup almost always).
    limit_entry_atr: float = 0.0

    # Regime / alignment gates
    use_regime_filter: bool = True
    require_mtf_alignment: bool = True
    alignment_threshold: float = 0.6
    regime_filter_penalty: float = field(default=1.5)  # extra score required when regime is poor

    # ICT setup-quality entry filters — the "trade only the real footprint"
    # layer. Measured on 18mo real XAUUSD: on H1 the winners are bar-momentum
    # confirmation + strict higher-timeframe direction. On M1 the sweep+reversal
    # filter cuts overtrading hard (but is neutral-to-negative on H1).
    require_sweep_reversal: bool = True
    sweep_reversal_window: int = 12
    require_entry_momentum: bool = True
    strict_mtf_direction: bool = True
    # Deprecated: was "spread in points" but the gate compared it to a PRICE
    # (quote_spread), so it could never trigger correctly. Use ``max_spread``
    # (PRICE units) instead — the scorer and the gate now share that one field.
    max_spread_points: float | None = None

    # Dynamic gate model (score-weighted executability). When ON, the auxiliary
    # quality gates no longer hard-stop an entry: each FAILED gate subtracts
    # ``gate_penalty`` points from the raw confluence score, and the bot
    # executes when (raw_score - penalties) >= threshold. A very strong setup
    # (score 6-7) therefore survives a minor gate miss; a marginal setup (score
    # at threshold) still needs every gate green. The spread gate is a COST
    # control and stays a hard block even in dynamic mode.
    dynamic_gates: bool = False
    gate_penalty: float = 1.0
    # Directional asymmetry: trade ONLY with the higher-timeframe macro trend.
    # When set (e.g. "d1"), longs require htf_<tf>_trend_strength > 0 and shorts
    # require < 0 — so in a confirmed bull market the strategy stops fading the
    # trend with counter-trend shorts (the main bleed in the old results). The
    # column is causal (pre-computed on the HTF, merged backward). Off by default
    # in the dataclass; enabled for backtests via configs/risk.yaml.
    htf_trend_timeframe: str | None = None

    # SMC microstructure score weights (see ICTConfluenceStrategy). 0 disables.
    # Measured on 2y real XAUUSD: scoring these into the threshold overtrades
    # and dilutes PF, so the validated default is OFF for the rule-based entry;
    # the features still reach the RL and the MTF context (htf_* columns).
    smc_displacement: int = 0
    smc_ifvg: int = 0
    smc_breaker: int = 0
    smc_vi: int = 0
    smc_dol_tap: int = 0


class PersonalityAdaptiveStrategy:
    """ICT confluence strategy whose behavior adapts to the market regime.

    Thresholds shift with volatility, trend and session; direction is gated by
    the trader's regime preferences; position size is risk-budgeted instead of
    fixed. The strategy never executes orders - it only emits OrderIntents.
    """

    name = "personality-adaptive"

    def __init__(
        self,
        personality: TraderPersonality | None = None,
        config: PersonalityAdaptiveConfig | None = None,
        *,
        symbol: str = "XAUUSD",
        volume: float = 0.1,
    ) -> None:
        if volume <= 0:
            raise ValueError("volume must be positive")
        self.symbol = symbol
        self.volume = volume
        self.personality = personality or TraderPersonality.from_yaml()
        self.config = config or PersonalityAdaptiveConfig()

        self._scorer = ICTConfluenceStrategy(
            symbol=symbol,
            volume=volume,
            min_score=1,
            cooldown_bars=0,
            discount_threshold=-0.15,
            premium_threshold=0.15,
            min_abs_trend_strength=self.config.min_abs_trend_strength,
            allowed_sessions=self.config.allowed_sessions,
            require_fresh_quote=self.config.require_fresh_quote,
            max_spread=self.config.max_spread,
            min_tick_rate=self.config.min_tick_rate,
            smc_displacement=self.config.smc_displacement,
            smc_ifvg=self.config.smc_ifvg,
            smc_breaker=self.config.smc_breaker,
            smc_vi=self.config.smc_vi,
            smc_dol_tap=self.config.smc_dol_tap,
        )
        self._context_engine = MarketContextEngine(self.personality, MarketRegimeEngine())
        self._alignment_engine = MicroMacroAlignmentEngine(self.personality)

        # Cheap scalar ring buffers for the causal context computation. The old
        # path rebuilt a (history_window x ~200 cols) DataFrame *twice* per bar
        # and re-classified the whole regime window per bar, which is O(window)
        # per bar — over a 2-year aligned frame that is both slow and memory
        # hungry. The buffers below keep only the scalars the context actually
        # reads, and the context itself is computed from numpy arrays in O(1).
        lookback = getattr(self._context_engine.regime_engine, "volatility_lookback", 100)
        self._window_maxlen = max(self.config.history_window, lookback)
        self._atr_norm: deque = deque(maxlen=self._window_maxlen)
        self._trend: deque = deque(maxlen=self._window_maxlen)
        self._premium: deque = deque(maxlen=self._window_maxlen)
        self._times: deque = deque(maxlen=self._window_maxlen)
        self._mtf_bias: deque = deque(maxlen=self._window_maxlen)
        self._mtf_conf: deque = deque(maxlen=self._window_maxlen)
        self._cols_initialized = False
        self._has_htf = False
        self._has_mtf_bias = False
        self._has_mtf_conf = False

        # Setup-quality ring buffers (the sweep/reversal entry filter reads a
        # short, causal lookback of the footprint columns).
        setup_window = max(self.config.sweep_reversal_window, 2)
        self._sweeps: deque = deque(maxlen=setup_window)
        self._bos: deque = deque(maxlen=setup_window)
        self._choch: deque = deque(maxlen=setup_window)

        self._side: str = "flat"
        self._last_entry_index: int = -1_000_000
        self._trades: list[OrderIntent] = []
        # Dynamic-gate observability (read by the live-loop decision trace).
        self._last_gate_penalty: float = 0.0
        self._last_effective_score: float = 0.0
        self._last_threshold: float = 0.0

    # ------------------------------------------------------------------
    # Public API (used by backtest engines)
    # ------------------------------------------------------------------
    def on_bar(self, index: int, bar: pd.Series) -> OrderIntent | None:
        """Return an OrderIntent when the persona wants to enter, else None."""
        self._push_history(bar)

        if index - self._last_entry_index < self.config.cooldown_bars:
            return None
        if self.config.require_fresh_quote and not bool(bar.get("quote_is_fresh", True)):
            return None
        if not _session_allowed(bar, self.config.allowed_sessions):
            return None

        context = self._context_engine.analyze_tail_arrays(
            atr_norm=np.asarray(list(self._atr_norm), dtype=float),
            trend_strength=np.asarray(list(self._trend), dtype=float),
            premium_discount=np.asarray(list(self._premium), dtype=float),
            times=list(self._times),
            mtf_bias=np.asarray(list(self._mtf_bias), dtype=float) if self._has_mtf_bias else None,
            mtf_confluence=np.asarray(list(self._mtf_conf), dtype=float) if self._has_mtf_conf else None,
            has_htf=self._has_htf,
        )
        alignment = self._alignment_engine.evaluate(None, context)

        # Personality-adaptive threshold -------------------------------
        threshold = self._adaptive_threshold(context)

        # Poor regime: demand a stronger setup (adaptive discipline)
        if float(context.get("regime_score", 0.5)) < 0.5:
            threshold = int(round(threshold + self.config.regime_filter_penalty))

        long_score = self._scorer._long_score(bar)  # type: ignore[attr-defined]
        short_score = self._scorer._short_score(bar)  # type: ignore[attr-defined]

        intent: OrderIntent | None = None
        if self.config.dynamic_gates:
            # DYNAMIC (score-weighted) executability: the bot decides whether a
            # setup is tradeable from its confluence score minus the penalties
            # of any failed auxiliary gate — a strong setup overrides minor
            # misses instead of being hard-stopped.
            if long_score >= short_score:
                side, raw_score = Side.BUY, long_score
            else:
                side, raw_score = Side.SELL, short_score
            if raw_score < 1 or (side == Side.BUY and self._side == "long") or (side == Side.SELL and self._side == "short"):
                return None
            penalty = self._gate_penalties(side, bar, context, alignment)
            self._last_gate_penalty = penalty
            self._last_effective_score = raw_score - penalty
            self._last_threshold = threshold
            if penalty >= float("inf"):
                return None  # spread cost-control remains a hard block
            if (raw_score - penalty) < threshold:
                return None
            intent = self._build_intent(side, raw_score, bar)
        else:
            # Legacy HARD gates (unchanged path, kept for A/B comparison).
            if self.config.use_regime_filter:
                if context["volatility"] not in self.personality.volatility_preferences:
                    return None
                if context["trend"] not in self.personality.trend_preferences:
                    return None
                if context["session"] not in self.personality.session_preferences:
                    return None
            if self.config.require_mtf_alignment and context.get("has_htf", False) and alignment < self.config.alignment_threshold:
                return None
            if long_score >= threshold and long_score > short_score and self._side != "long":
                intent = self._build_intent(Side.BUY, long_score, bar)
            elif short_score >= threshold and short_score > long_score and self._side != "short":
                intent = self._build_intent(Side.SELL, short_score, bar)
            if intent is not None and not self._setup_quality_ok(intent.side, bar):
                intent = None

        if intent is not None:
            self._side = "long" if intent.side == Side.BUY else "short"
            self._last_entry_index = index
            self._trades.append(intent)
        return intent

    def on_position_closed(self) -> None:
        """Called by the managed-exit engine when a managed trade fully closes.

        The engine manages position state (SL/TP/trailing) and only requests a
        new entry while flat — but this strategy tracked its own ``_side`` and
        never learned the exit happened, which locked it out of same-direction
        re-entries. Resetting to flat lets the next bias-aligned setup enter.
        """
        self._side = "flat"

    def reset(self) -> None:
        """Reset in-strategy state (used between episodes/walk-forward folds)."""
        self._atr_norm.clear()
        self._trend.clear()
        self._premium.clear()
        self._times.clear()
        self._mtf_bias.clear()
        self._mtf_conf.clear()
        self._sweeps.clear()
        self._bos.clear()
        self._choch.clear()
        self._side = "flat"
        self._last_entry_index = -1_000_000
        self._trades.clear()
        self._cols_initialized = False

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def warm_context(self, bars: pd.DataFrame) -> None:
        """Populate the regime/alignment context from recent featured bars.

        Used by the live loop at startup so its FIRST decision (on the last
        closed bar) is as faithful as a mid-session bar-close decision. Feeds
        ``_push_history`` for each row WITHOUT emitting orders or touching
        ``_side`` / ``_last_entry_index`` — it only warms the rolling context.
        """
        for _, row in bars.iterrows():
            self._push_history(row)

    def _push_history(self, bar: pd.Series) -> None:
        """Buffer the scalars the context engine reads, in causal order.

        Mirrors the old ``self._history.append(bar.copy())`` behaviour (every
        bar is buffered, even during cooldown / non-fresh bars), but stores
        only the handful of values the regime + context computation needs.
        """
        if not self._cols_initialized:
            self._has_mtf_bias = "mtf_bias" in bar.index
            self._has_mtf_conf = "mtf_confluence_score" in bar.index
            self._has_htf = self._has_mtf_bias or self._has_mtf_conf
            self._cols_initialized = True
        self._atr_norm.append(_to_float(bar.get("atr_norm", 0.0)))
        self._trend.append(_to_float(bar.get("trend_strength", 0.0)))
        self._premium.append(_to_float(bar.get("premium_discount", 0.0)))
        self._times.append(bar.get("time", None))
        if self._has_mtf_bias:
            self._mtf_bias.append(_to_float(bar.get("mtf_bias", 0.0)))
        if self._has_mtf_conf:
            self._mtf_conf.append(_to_float(bar.get("mtf_confluence_score", 0.0)))
        self._sweeps.append(_to_float(bar.get("liquidity_sweep", 0.0)))
        self._bos.append(_to_float(bar.get("bos_dir", 0.0)))
        self._choch.append(_to_float(bar.get("choch_dir", 0.0)))

    def _setup_quality_ok(self, side: Side, bar: pd.Series) -> bool:
        """Return True when the entry is a high-quality ICT footprint.

        The checks are cheap and causal: they only read the current bar plus
        the short setup ring buffers (all <= the current bar).
        """
        cfg = self.config
        long = side == Side.BUY

        # Momentum: enter WITH the bar's own direction (no fading a candle).
        if cfg.require_entry_momentum:
            open_ = _to_float(bar.get("open", 0.0))
            close_ = _to_float(bar.get("close", 0.0))
            if open_ and close_:
                if long and close_ <= open_:
                    return False
                if not long and close_ >= open_:
                    return False

        # Spread gate: never pay a wide spread for a setup. Both the scorer and
        # this gate share the PRICE-unit threshold (``max_spread``); the old
        # ``max_spread_points`` compared price against points, which could never
        # trigger correctly. A missing column / None threshold = gate off.
        if cfg.max_spread is not None:
            if _to_float(bar.get("quote_spread", 0.0)) > cfg.max_spread:
                return False

        # Strict MTF direction: trade WITH higher-timeframe structure only.
        if cfg.strict_mtf_direction and self._has_mtf_bias and self._mtf_bias:
            bias = self._mtf_bias[-1]
            if long and bias < 0:
                return False
            if not long and bias > 0:
                return False

        # HTF macro-trend asymmetry: trade only in the D1 trend's direction.
        # Guarded on column presence so live bars without htf_* columns are
        # unaffected (the gate is then a no-op, not a full block).
        if cfg.htf_trend_timeframe:
            trend_col = f"htf_{cfg.htf_trend_timeframe}_trend_strength"
            if trend_col in bar.index:
                ts = _to_float(bar.get(trend_col, 0.0))
                if long and ts <= 0:
                    return False
                if not long and ts >= 0:
                    return False

        # Sweep + reversal: the actual ICT setup — a liquidity sweep that has
        # been confirmed by a structure shift in the reversal's direction.
        if cfg.require_sweep_reversal:
            sweeps = list(self._sweeps)
            bos = list(self._bos)
            choch = list(self._choch)
            if not sweeps:
                return False
            if long:
                swept = any(s < 0 for s in sweeps)
                structure = any(b > 0 or c > 0 for b, c in zip(bos, choch, strict=False))
            else:
                swept = any(s > 0 for s in sweeps)
                structure = any(b < 0 or c < 0 for b, c in zip(bos, choch, strict=False))
            if not (swept and structure):
                return False

        return True

    def _gate_penalties(self, side: Side, bar: pd.Series, context: dict, alignment: float) -> float:
        """Dynamic gate model: score-weighted executability.

        Instead of hard-stopping an entry when one auxiliary check fails, each
        failed gate subtracts ``gate_penalty`` from the raw confluence score.
        Returns a non-negative penalty; ``float("inf")`` means a hard block
        (only used for the spread cost-control, which must never be overridden).
        """
        cfg = self.config
        long = side == Side.BUY
        penalty = 0.0

        # Regime preference filter — soft: an unfavourable regime raises the
        # bar instead of banning the entry outright.
        if cfg.use_regime_filter:
            if context["volatility"] not in self.personality.volatility_preferences:
                penalty += cfg.gate_penalty
            if context["trend"] not in self.personality.trend_preferences:
                penalty += cfg.gate_penalty
            if context["session"] not in self.personality.session_preferences:
                penalty += cfg.gate_penalty

        # MTF alignment — soft: misalignment against higher timeframes costs
        # points instead of discarding a strong local setup.
        if cfg.require_mtf_alignment and context.get("has_htf", False) and alignment < cfg.alignment_threshold:
            penalty += cfg.gate_penalty

        # Momentum — soft: a strong confluence score may override a candle
        # fading the signal.
        if cfg.require_entry_momentum:
            open_ = _to_float(bar.get("open", 0.0))
            close_ = _to_float(bar.get("close", 0.0))
            if open_ and close_ and ((long and close_ <= open_) or (not long and close_ >= open_)):
                penalty += cfg.gate_penalty

        # Spread — HARD: a wide spread is a cost reality, not a quality call.
        if cfg.max_spread is not None:
            if _to_float(bar.get("quote_spread", 0.0)) > cfg.max_spread:
                return float("inf")

        # Strict MTF direction — soft.
        if cfg.strict_mtf_direction and self._has_mtf_bias and self._mtf_bias:
            bias = self._mtf_bias[-1]
            if (long and bias < 0) or (not long and bias > 0):
                penalty += cfg.gate_penalty

        # HTF macro-trend asymmetry — soft.
        if cfg.htf_trend_timeframe:
            trend_col = f"htf_{cfg.htf_trend_timeframe}_trend_strength"
            if trend_col in bar.index:
                ts = _to_float(bar.get(trend_col, 0.0))
                if (long and ts <= 0) or (not long and ts >= 0):
                    penalty += cfg.gate_penalty

        # Sweep + reversal — soft.
        if cfg.require_sweep_reversal:
            sweeps = list(self._sweeps)
            bos = list(self._bos)
            choch = list(self._choch)
            if not sweeps:
                penalty += cfg.gate_penalty
            else:
                if long:
                    swept = any(s < 0 for s in sweeps)
                    structure = any(b > 0 or c > 0 for b, c in zip(bos, choch, strict=False))
                else:
                    swept = any(s > 0 for s in sweeps)
                    structure = any(b < 0 or c < 0 for b, c in zip(bos, choch, strict=False))
                if not (swept and structure):
                    penalty += cfg.gate_penalty

        return penalty

    def _adaptive_threshold(self, context: dict) -> int:
        """Compute the entry-score threshold from personality + regime.

        Higher selectivity/conviction & worse regimes -> higher threshold.
        Higher aggression/adaptability & favorable regimes -> lower threshold.
        """
        personality = self.personality
        base = float(personality.confidence_thresholds.get("min_entry_score", self.config.min_score))

        # Selectivity raises the bar; aggression lowers it.
        threshold = base + (personality.selectivity - 0.5) * 2.0 - (personality.aggression - 0.5) * 2.0

        # Edge optimism: skeptical personas need stronger confirmations.
        threshold += (0.5 - personality.edge_optimism) * 2.0

        # Regime shifts
        volatility = context.get("volatility", "normal")
        if volatility == "high":
            threshold += 1.0 * personality.adaptability  # high vol -> be picky
        elif volatility == "low":
            threshold -= 1.0 * personality.adaptability  # low vol -> easier edges
        if context.get("macro_strength") == "weak":
            threshold += 1.0 * personality.structure_focus

        # Session boost
        if context.get("session") in personality.session_preferences:
            threshold -= 0.5 * personality.session_sensitivity

        return max(1, int(round(threshold)))

    def _build_intent(self, side: Side, score: int, bar: pd.Series) -> OrderIntent:
        """Create the OrderIntent, applying risk-based sizing when enabled."""
        reason = f"persona_{side.value}_{score}"
        if self.config.risk_based_sizing:
            volume = self._risk_sized_volume(bar)
        else:
            volume = self.volume
        kind = OrderKind.MARKET
        limit_price: float | None = None
        if self.config.limit_entry_atr and self.config.limit_entry_atr > 0:
            atr = float(bar.get("atr", 0.0) or 0.0)
            close = float(bar.get("close", 0.0) or 0.0)
            if atr > 0 and close > 0:
                offset = self.config.limit_entry_atr * atr
                kind = OrderKind.LIMIT
                limit_price = round(close - offset, 5) if side == Side.BUY else round(close + offset, 5)
        return OrderIntent(symbol=self.symbol, side=side, volume=volume, reason=reason, kind=kind, limit_price=limit_price)

    def _risk_sized_volume(self, bar: pd.Series) -> float:
        """Size so a stop-out loses approximately risk_per_trade of equity."""
        atr = float(bar.get("atr", 0.0) or 0.0)
        stop_distance = max(atr * self.config.stop_loss_atr, self.config.min_slot_distance_price)
        risk_budget = self.config.equity * self.config.risk_per_trade
        denominator = stop_distance * self.config.point_value
        if denominator <= 0:
            return self.volume
        volume = risk_budget / denominator
        if volume <= 0 or not volume == volume:  # NaN guard
            return self.volume
        return round(float(volume), 8)


def _to_float(value: Any) -> float:
    """Coerce a bar scalar to float, mapping missing/NaN to 0.0.

    Mirrors ``MarketRegimeEngine._column_or_zeros`` so the fast context path
    sees the same numbers the old DataFrame path produced.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if result != result else result


def _session_allowed(bar: pd.Series, allowed_sessions: tuple[str, ...]) -> bool:
    if not allowed_sessions:
        return True
    session_columns = {
        "asia": "session_asia",
        "london": "session_london",
        "ny_am": "session_ny_am",
        "ny_pm": "session_ny_pm",
        "other": "session_other",
    }
    present = [column for column in session_columns.values() if column in bar.index]
    if not present:
        return True
    for session in allowed_sessions:
        column = session_columns.get(session)
        if column and float(bar.get(column, 0.0)) > 0:
            return True
    return False
