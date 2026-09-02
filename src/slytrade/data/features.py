"""Per-timeframe feature engineering for ICT/SMC scalping.

Every feature computed here is strictly causal: the value at row ``t``
depends only on bars ``[0 .. t]`` -- never on future bars.  This makes
features safe to use in both backtest (Layer 5) and live trading.

Processing is per-TF only (no cross-TF alignment here -- that is Layer 3).

Output columns (grouped):

    Basics / candle math:
        body, range_, upper_wick, lower_wick
        body_pct, upper_wick_pct, lower_wick_pct
        direction (+1/-1/0), close_location (0..1)
        bull_engulf, bear_engulf
        body_avg_ratio (body / SMA(body,20))

    Volatility (Wilder ATR):
        atr_14, atr_pct (atr/close), atr_expand (atr/SMA(atr,50))
        bar_range_atr (range/atr)

    EMAs:
        ema_20, ema_50, ema_200
        ema20_above_ema50, ema50_above_ema200
        price_above_ema20, price_above_ema50, price_above_ema200
        ema_slope_20 ((ema[t]-ema[t-1])/atr)

    Swings & structure (BOS / CHoCH):
        Two swing sizes -- minor (short-term entry structure) and major
        (longer-term bias structure).  Default detector is
        ``atr_zigzag``: a swing high/low is confirmed only after price
        has retraced ``mult * ATR`` away from the extreme (with a
        2-bar minimum gap).  This is TF-agnostic and volatility-adaptive
        -- the same parameters produce structurally meaningful swings on
        every timeframe from M1 to W1 and automatically adapt to session
        volatility (wider in London/NY, tighter in Asia).
        minor_swing_high, minor_swing_low (price of last confirmed swing)
        major_swing_high, major_swing_low
        minor_bias (+1 bull / -1 bear / 0)
        major_bias
        minor_bos_up, minor_bos_dn, minor_choch_up, minor_choch_dn
        major_bos_up, major_bos_dn, major_choch_up, major_choch_dn

        ``swing_method='fractal'`` selects a legacy fixed-bar fractal
        (``swing_minor`` / ``swing_major`` bars each side of the pivot)
        for comparison/debugging.

    Displacement:
        bull_disp, bear_disp (full-body impulsive candle)
        vol_spike (tick_volume > 2x 20-bar SMA)

    Liquidity sweeps (stop-runs before reversal):
        bull_liq_sweep — wick takes out a recent minor swing LOW (sellside
            liquidity), then closes BACK ABOVE it (rejection / trap). The
            sweep wick extreme (the liquidity level) is recorded for SL.
        bear_liq_sweep — symmetric: wick takes out a recent minor swing
            HIGH (buyside liq), closes back BELOW it.
        bull_sweep_px, bear_sweep_px — extreme price of the most recent
            un-reclaimed sweep wick (SL anchor), NaN once price closes
            beyond the sweep level (invalidation / BOS instead).

    Order Blocks (last opposite-colour close before displacement):
        bull_ob_top, bull_ob_bottom, bull_ob_idx, bull_ob_mitigated
        bear_ob_top, bear_ob_bottom, bear_ob_idx, bear_ob_mitigated
        Only the single most-recent unmitigated OB per side is tracked.

    Fair Value Gaps (3-candle imbalance):
        bull_fvg_top, bull_fvg_bottom, bull_fvg_idx, bull_fvg_mitigated
        bear_fvg_top, bear_fvg_bottom, bear_fvg_idx, bear_fvg_mitigated

    Premium / Discount within current major swing range:
        range_high, range_low
        price_in_range_pct
        is_premium (>0.67), is_discount (<0.33), is_equilibrium (mid)

    Sessions / killzones (UTC -- works for XAUUSD):
        session (ASIA/LONDON/NY/OFF), hour, minute, dow
        kz_asian (00-03 UTC), kz_london (07-10 UTC), kz_ny (12-15 UTC)
        london_open_30 (08:00-08:30 UTC), ny_open_30 (13:30-14:00 UTC)

    Volume:
        tick_vol_sma20, tick_vol_ratio
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureConfig:
    """Tunables for per-TF feature processing.

    Swing detection defaults to ``atr_zigzag`` which is TF-agnostic and
    volatility-adaptive (recommended): a swing is confirmed once price
    CLOSES more than ``swing_minor_atr_mult`` * ATR (for minor) or
    ``swing_major_atr_mult`` * ATR (for major) away from the most recent
    opposite extreme, with at least ``swing_min_bars`` bars between
    extreme and confirmation.  This produces structurally consistent
    swings on M1..W1 without hand-tuned per-TF lookbacks, and
    auto-adjusts when volatility expands/compresses (wider swings in
    London/NY, tighter swings in Asian chop).

    ``swing_method='fractal'`` selects the legacy fixed-bar fractal
    (``swing_minor`` / ``swing_major`` bars each side of the pivot).
    """
    atr_len: int = 14
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    atr_smooth_len: int = 50
    body_sma_len: int = 20
    vol_sma_len: int = 20
    # Swing detection -- ATR-ZigZag (default, adaptive).
    swing_method: str = "atr_zigzag"        # "atr_zigzag" | "fractal"
    swing_minor_atr_mult: float = 1.5       # ~1.5 ATR retrace = minor
    swing_major_atr_mult: float = 4.0       # ~4.0 ATR retrace = major
    swing_min_bars: int = 2                 # min bars extreme -> confirmation
    # Legacy fixed-bar fractal lookbacks (used when swing_method='fractal').
    swing_minor: int = 2
    swing_major: int = 5
    # Displacement / volume.
    disp_body_pct: float = 0.6
    disp_close_loc: float = 0.75
    disp_atr_mult: float = 1.5
    vol_spike_mult: float = 2.0


DEFAULT_CONFIG = FeatureConfig()


# --------------------------------------------------------------------------- #
# Session / killzone helpers (UTC based)
# --------------------------------------------------------------------------- #
def _add_sessions(df: pd.DataFrame) -> None:
    """Add session/killzone columns. Mutates df in place."""
    ts = df["time"].dt
    h = ts.hour.astype(np.int32)
    m = ts.minute.astype(np.int32)
    dow = ts.dayofweek.astype(np.int32)
    hm = h * 60 + m  # minutes into UTC day

    conditions = [
        (h < 8),
        (h >= 8) & (h < 13),
        (h >= 13) & (h < 17),
    ]
    choices = ["ASIA", "LONDON", "NY"]
    df["session"] = np.select(conditions, choices, default="OFF")

    df["hour"] = h.astype(np.int16)
    df["minute"] = m.astype(np.int16)
    df["dow"] = dow.astype(np.int16)

    df["kz_asian"] = ((hm >= 0) & (hm < 180)).astype(bool)
    df["kz_london"] = ((hm >= 420) & (hm < 600)).astype(bool)
    df["kz_ny"] = ((hm >= 720) & (hm < 900)).astype(bool)

    df["london_open_30"] = ((hm >= 480) & (hm < 510)).astype(bool)
    df["ny_open_30"] = ((hm >= 810) & (hm < 840)).astype(bool)


# --------------------------------------------------------------------------- #
# Candle basics
# --------------------------------------------------------------------------- #
def _add_basics(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0.0, np.nan)
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    df["body"] = body.astype(np.float64)
    df["range_"] = (h - l).astype(np.float64)
    df["upper_wick"] = upper_wick.astype(np.float64)
    df["lower_wick"] = lower_wick.astype(np.float64)
    df["body_pct"] = (body / rng).fillna(0.0).clip(0.0, 1.0).astype(np.float64)
    df["upper_wick_pct"] = (upper_wick / rng).fillna(0.0).clip(0.0, 1.0).astype(np.float64)
    df["lower_wick_pct"] = (lower_wick / rng).fillna(0.0).clip(0.0, 1.0).astype(np.float64)
    df["direction"] = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(np.int8)

    df["close_location"] = ((c - l) / rng).fillna(0.5).clip(0.0, 1.0).astype(np.float64)

    # Engulfing.
    prev_body_high = pd.concat([df["open"].shift(1), df["close"].shift(1)], axis=1).max(axis=1)
    prev_body_low = pd.concat([df["open"].shift(1), df["close"].shift(1)], axis=1).min(axis=1)
    body_high = pd.concat([o, c], axis=1).max(axis=1)
    body_low = pd.concat([o, c], axis=1).min(axis=1)
    prev_dir = df["direction"].shift(1)
    df["bull_engulf"] = (
        (df["direction"] == 1)
        & (prev_dir == -1)
        & (body_low < prev_body_low)
        & (body_high > prev_body_high)
    ).fillna(False).astype(bool)
    df["bear_engulf"] = (
        (df["direction"] == -1)
        & (prev_dir == 1)
        & (body_high > prev_body_high)
        & (body_low < prev_body_low)
    ).fillna(False).astype(bool)

    body_sma = body.rolling(cfg.body_sma_len, min_periods=1).mean()
    df["body_avg_ratio"] = (body / body_sma.replace(0.0, np.nan)).fillna(1.0).astype(np.float64)


# --------------------------------------------------------------------------- #
# ATR (Wilder / RMA), EMAs
# --------------------------------------------------------------------------- #
def _wilder(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (same as TradingView RMA / MT5 iATR)."""
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()


def _add_volatility(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    h, l, c = df["high"], df["low"], df["close"]
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, cfg.atr_len)
    df["atr_14"] = atr.astype(np.float64)
    df["atr_pct"] = (atr / c.replace(0.0, np.nan)).fillna(0.0).astype(np.float64)
    atr_sma = atr.rolling(cfg.atr_smooth_len, min_periods=1).mean()
    df["atr_expand"] = (atr / atr_sma.replace(0.0, np.nan)).fillna(1.0).astype(np.float64)
    df["bar_range_atr"] = (df["range_"] / atr.replace(0.0, np.nan)).fillna(0.0).astype(np.float64)


def _add_emas(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    c = df["close"]
    ema20 = c.ewm(span=cfg.ema_fast, adjust=False).mean()
    ema50 = c.ewm(span=cfg.ema_mid, adjust=False).mean()
    ema200 = c.ewm(span=cfg.ema_slow, adjust=False).mean()
    df["ema_20"] = ema20.astype(np.float64)
    df["ema_50"] = ema50.astype(np.float64)
    df["ema_200"] = ema200.astype(np.float64)
    df["ema20_above_ema50"] = (ema20 > ema50).fillna(False).astype(bool)
    df["ema50_above_ema200"] = (ema50 > ema200).fillna(False).astype(bool)
    df["price_above_ema20"] = (c > ema20).fillna(False).astype(bool)
    df["price_above_ema50"] = (c > ema50).fillna(False).astype(bool)
    df["price_above_ema200"] = (c > ema200).fillna(False).astype(bool)
    slope = (ema20 - ema20.shift(1)) / df["atr_14"].replace(0.0, np.nan)
    df["ema_slope_20"] = slope.fillna(0.0).astype(np.float64)


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
def _add_volume(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    tv = df["tick_volume"].astype(np.float64)
    vol_sma = tv.rolling(cfg.vol_sma_len, min_periods=1).mean()
    df["tick_vol_sma20"] = vol_sma.astype(np.float64)
    df["tick_vol_ratio"] = (tv / vol_sma.replace(0.0, np.nan)).fillna(1.0).astype(np.float64)
    df["vol_spike"] = (tv > cfg.vol_spike_mult * vol_sma).fillna(False).astype(bool)


# --------------------------------------------------------------------------- #
# Swing pivot detectors (causal)
#
# Both detectors return two int64 arrays of length ``n``:
#   sh[i] = index of the swing-high pivot *confirmed* at bar i, or -1
#   sl[i] = index of the swing-low  pivot *confirmed* at bar i, or -1
#
# "Confirmed at bar i" means the decision uses bars 0..i only (causal).
# The pivot bar itself (the extreme) can be any bar <= i.
# --------------------------------------------------------------------------- #
def _fractal_pivots(high: np.ndarray, low: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-bar fractal pivot detector (legacy).

    A pivot high at ``p`` requires ``high[p]`` strictly greater than
    the ``lookback`` bars to the left AND >= ``lookback`` bars to the
    right.  Confirmation at bar ``p + lookback``.
    """
    n = len(high)
    sh = np.full(n, -1, dtype=np.int64)
    sl = np.full(n, -1, dtype=np.int64)
    for p in range(lookback, n - lookback):
        left_h = high[p - lookback:p]
        right_h = high[p + 1:p + lookback + 1]
        if high[p] > left_h.max() and high[p] >= right_h.max():
            sh[p + lookback] = p
        left_l = low[p - lookback:p]
        right_l = low[p + 1:p + lookback + 1]
        if low[p] < left_l.min() and low[p] <= right_l.min():
            sl[p + lookback] = p
    return sh, sl


def _atr_zigzag_pivots(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    atr_mult: float,
    min_bars: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """ATR-adaptive ZigZag pivot detector (default).

    Single causal forward pass that mirrors how a human ICT trader
    draws structure by eye:
      * We track a running extreme high and extreme low as we sweep
        forward.
      * A swing HIGH is confirmed at bar ``i`` once ``close[i]`` is
        more than ``atr_mult * atr[i]`` below the running-extreme-high
        AND at least ``min_bars`` bars have elapsed since that extreme.
        The pivot bar is the bar of the extreme high.  After
        confirming, we switch to searching for the next swing low.
      * A swing LOW is confirmed symmetrically.
    """
    n = len(high)
    sh = np.full(n, -1, dtype=np.int64)
    sl = np.full(n, -1, dtype=np.int64)

    SEARCH_HIGH = 1   # leg is up from a confirmed low; tracking next high
    SEARCH_LOW = -1   # leg is down from a confirmed high; tracking next low
    UNCOMMITTED = 0

    state = UNCOMMITTED
    ext_hi_p = float(high[0])
    ext_hi_i = 0
    ext_lo_p = float(low[0])
    ext_lo_i = 0

    for i in range(1, n):
        h = float(high[i])
        l = float(low[i])
        c = float(close[i])
        a = float(atr[i])
        if not np.isfinite(a) or a <= 0.0:
            # ATR not yet warm -- just update running extremes.
            if h > ext_hi_p:
                ext_hi_p, ext_hi_i = h, i
            if l < ext_lo_p:
                ext_lo_p, ext_lo_i = l, i
            continue
        thresh = atr_mult * a

        if state == UNCOMMITTED:
            broke_up = c > ext_lo_p + thresh and (i - ext_lo_i) >= min_bars
            broke_dn = c < ext_hi_p - thresh and (i - ext_hi_i) >= min_bars
            if broke_up and not broke_dn:
                # First confirmed pivot: a LOW at ext_lo_i; now look for high.
                sl[i] = ext_lo_i
                state = SEARCH_HIGH
                ext_hi_p, ext_hi_i = h, i
                continue
            if broke_dn and not broke_up:
                # First confirmed pivot: a HIGH at ext_hi_i; now look for low.
                sh[i] = ext_hi_i
                state = SEARCH_LOW
                ext_lo_p, ext_lo_i = l, i
                continue
            # Neither (or both tied) -- keep tracking both extremes.
            if h > ext_hi_p:
                ext_hi_p, ext_hi_i = h, i
            if l < ext_lo_p:
                ext_lo_p, ext_lo_i = l, i
            continue

        if state == SEARCH_HIGH:
            # Up-leg from last confirmed low; update running high.
            if h > ext_hi_p:
                ext_hi_p, ext_hi_i = h, i
            # Swing high confirms when price falls back by threshold.
            if c < ext_hi_p - thresh and (i - ext_hi_i) >= min_bars:
                sh[i] = ext_hi_i
                state = SEARCH_LOW
                ext_lo_p, ext_lo_i = l, i
            continue

        if state == SEARCH_LOW:
            # Down-leg from last confirmed high; update running low.
            if l < ext_lo_p:
                ext_lo_p, ext_lo_i = l, i
            # Swing low confirms when price rallies back by threshold.
            if c > ext_lo_p + thresh and (i - ext_lo_i) >= min_bars:
                sl[i] = ext_lo_i
                state = SEARCH_HIGH
                ext_hi_p, ext_hi_i = h, i
            continue

    return sh, sl


# --------------------------------------------------------------------------- #
# Structure scan (BOS / CHoCH)
#
# Consumes pivot-confirmation arrays (sh, sl) from either detector and
# produces swing prices/indices, bias, BOS, and CHoCH columns.
#
# State machine (strictly causal):
#   * cur_sh_p / cur_sl_p hold the most recently confirmed swing prices.
#   * anchor_sh_p / anchor_sl_p are the structural levels that must
#     break for BOS (continuation) or CHoCH (trend failure).
#   * sh_broken / sl_broken edge-detect flags ensure a BOS/CHoCH event
#     fires on exactly ONE bar (the first bar that closes beyond the
#     anchor) rather than on every subsequent bar beyond it.
# --------------------------------------------------------------------------- #
def _structure_scan(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sh_conf: np.ndarray,
    sl_conf: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(high)
    out_sh_price = np.full(n, np.nan, dtype=np.float64)
    out_sl_price = np.full(n, np.nan, dtype=np.float64)
    out_sh_idx = np.full(n, -1, dtype=np.int64)
    out_sl_idx = np.full(n, -1, dtype=np.int64)
    bias = np.zeros(n, dtype=np.int8)
    bos_up = np.zeros(n, dtype=bool)
    bos_dn = np.zeros(n, dtype=bool)
    choch_up = np.zeros(n, dtype=bool)
    choch_dn = np.zeros(n, dtype=bool)

    cur_sh_p: float | None = None
    cur_sh_i: int = -1
    cur_sl_p: float | None = None
    cur_sl_i: int = -1

    anchor_sh_p: float | None = None
    anchor_sl_p: float | None = None
    cur_bias: int = 0

    # Edge-detect: a given anchor can be broken exactly once.
    sh_broken = False
    sl_broken = False

    for i in range(n):
        # 1. Register any pivot confirmed at this bar.
        if sh_conf[i] >= 0:
            p = int(sh_conf[i])
            cur_sh_p = float(high[p])
            cur_sh_i = p
            # A new swing high at/above the bear anchor resets that
            # side's "broken" flag so a future break of this new level
            # fires a fresh CHoCH/BOS.
            if anchor_sh_p is None or cur_sh_p >= anchor_sh_p:
                sh_broken = False
        if sl_conf[i] >= 0:
            p = int(sl_conf[i])
            cur_sl_p = float(low[p])
            cur_sl_i = p
            if anchor_sl_p is None or cur_sl_p <= anchor_sl_p:
                sl_broken = False

        c = float(close[i])
        new_bos_up = False
        new_bos_dn = False
        new_choch_up = False
        new_choch_dn = False

        # 2. Initial bias: once we have both an SH and SL, the first
        #    close outside the [SL, SH] range sets initial direction.
        if cur_bias == 0:
            if cur_sh_p is not None and cur_sl_p is not None:
                if c > cur_sh_p and not sh_broken:
                    new_bos_up = True
                    cur_bias = 1
                    anchor_sh_p = cur_sh_p
                    anchor_sl_p = cur_sl_p
                    sh_broken = True
                elif c < cur_sl_p and not sl_broken:
                    new_bos_dn = True
                    cur_bias = -1
                    anchor_sh_p = cur_sh_p
                    anchor_sl_p = cur_sl_p
                    sl_broken = True
        elif cur_bias == 1:
            # Bullish.  CHoCH dn: close < anchor SL (failure).
            if anchor_sl_p is not None and c < anchor_sl_p and not sl_broken:
                new_choch_dn = True
                cur_bias = -1
                anchor_sh_p = cur_sh_p
                anchor_sl_p = cur_sl_p
                sh_broken = False
                sl_broken = True   # old bull SL just broken
            elif anchor_sh_p is not None and c > anchor_sh_p and not sh_broken:
                new_bos_up = True
                # Advance bull-leg anchors to latest confirmed swings.
                anchor_sh_p = cur_sh_p if cur_sh_p is not None else anchor_sh_p
                anchor_sl_p = cur_sl_p if cur_sl_p is not None else anchor_sl_p
                sh_broken = True
        elif cur_bias == -1:
            # Bearish.  CHoCH up: close > anchor SH (failure).
            if anchor_sh_p is not None and c > anchor_sh_p and not sh_broken:
                new_choch_up = True
                cur_bias = 1
                anchor_sh_p = cur_sh_p
                anchor_sl_p = cur_sl_p
                sh_broken = True
                sl_broken = False
            elif anchor_sl_p is not None and c < anchor_sl_p and not sl_broken:
                new_bos_dn = True
                anchor_sl_p = cur_sl_p if cur_sl_p is not None else anchor_sl_p
                anchor_sh_p = cur_sh_p if cur_sh_p is not None else anchor_sh_p
                sl_broken = True

        bos_up[i] = new_bos_up
        bos_dn[i] = new_bos_dn
        choch_up[i] = new_choch_up
        choch_dn[i] = new_choch_dn
        bias[i] = cur_bias

        out_sh_price[i] = cur_sh_p if cur_sh_p is not None else np.nan
        out_sl_price[i] = cur_sl_p if cur_sl_p is not None else np.nan
        out_sh_idx[i] = cur_sh_i
        out_sl_idx[i] = cur_sl_i

    return {
        "swing_high": out_sh_price,
        "swing_low": out_sl_price,
        "swing_high_idx": out_sh_idx,
        "swing_low_idx": out_sl_idx,
        "bias": bias,
        "bos_up": bos_up,
        "bos_dn": bos_dn,
        "choch_up": choch_up,
        "choch_dn": choch_dn,
    }


def _add_structure(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    atr = df["atr_14"].to_numpy(dtype=np.float64)

    tiers = [
        ("minor", cfg.swing_minor_atr_mult, cfg.swing_min_bars, cfg.swing_minor),
        ("major", cfg.swing_major_atr_mult, cfg.swing_min_bars, cfg.swing_major),
    ]

    for label, atr_mult, min_bars, lb in tiers:
        if cfg.swing_method == "fractal":
            sh, sl = _fractal_pivots(h, l, lb)
        else:
            sh, sl = _atr_zigzag_pivots(h, l, c, atr, atr_mult, min_bars=min_bars)

        res = _structure_scan(h, l, c, sh, sl)
        df[f"{label}_swing_high"] = res["swing_high"]
        df[f"{label}_swing_low"] = res["swing_low"]
        df[f"{label}_swing_high_idx"] = res["swing_high_idx"]
        df[f"{label}_swing_low_idx"] = res["swing_low_idx"]
        df[f"{label}_bias"] = res["bias"].astype(np.int8)
        df[f"{label}_bos_up"] = res["bos_up"].astype(bool)
        df[f"{label}_bos_dn"] = res["bos_dn"].astype(bool)
        df[f"{label}_choch_up"] = res["choch_up"].astype(bool)
        df[f"{label}_choch_dn"] = res["choch_dn"].astype(bool)


# --------------------------------------------------------------------------- #
# Displacement
# --------------------------------------------------------------------------- #
def _add_displacement(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    bull_disp = (
        (df["direction"] == 1)
        & (df["body_pct"] >= cfg.disp_body_pct)
        & (df["close_location"] >= cfg.disp_close_loc)
        & (df["bar_range_atr"] >= cfg.disp_atr_mult)
    )
    bear_disp = (
        (df["direction"] == -1)
        & (df["body_pct"] >= cfg.disp_body_pct)
        & (df["close_location"] <= (1.0 - cfg.disp_close_loc))
        & (df["bar_range_atr"] >= cfg.disp_atr_mult)
    )
    df["bull_disp"] = bull_disp.fillna(False).astype(bool)
    df["bear_disp"] = bear_disp.fillna(False).astype(bool)


# --------------------------------------------------------------------------- #
# Order Blocks
# --------------------------------------------------------------------------- #
def _add_order_blocks(df: pd.DataFrame) -> None:
    """Most-recent unmitigated OB per side.

    Bullish OB: last bearish candle before a bullish displacement candle.
        - top = OB high, bottom = OB low
        - mitigated when a subsequent WICK trades below OB bottom
          (low[i] < bot) — wick-taps invalidate, not just closes.
    """
    n = len(df)
    direction = df["direction"].to_numpy()
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    bull_disp = df["bull_disp"].to_numpy()
    bear_disp = df["bear_disp"].to_numpy()

    bull_ob_top = np.full(n, np.nan, dtype=np.float64)
    bull_ob_bottom = np.full(n, np.nan, dtype=np.float64)
    bull_ob_idx = np.full(n, -1, dtype=np.int64)
    bull_ob_mit = np.zeros(n, dtype=bool)
    bear_ob_top = np.full(n, np.nan, dtype=np.float64)
    bear_ob_bottom = np.full(n, np.nan, dtype=np.float64)
    bear_ob_idx = np.full(n, -1, dtype=np.int64)
    bear_ob_mit = np.zeros(n, dtype=bool)

    last_bear_idx = -1
    last_bull_idx = -1

    act_bull_top: float | None = None
    act_bull_bot: float | None = None
    act_bull_i: int = -1
    act_bear_top: float | None = None
    act_bear_bot: float | None = None
    act_bear_i: int = -1

    for i in range(n):
        # Wick-based mitigation: low[i] < bull_bot => bull OB mitigated;
        # high[i] > bear_top => bear OB mitigated. Catches wicks that close
        # back inside (common during news/London spikes).
        if act_bull_top is not None and low[i] < act_bull_bot:
            act_bull_top = act_bull_bot = None
            act_bull_i = -1
        if act_bear_bot is not None and high[i] > act_bear_top:
            act_bear_top = act_bear_bot = None
            act_bear_i = -1

        if bull_disp[i] and last_bear_idx >= 0:
            ob_i = last_bear_idx
            act_bull_top = float(high[ob_i])
            act_bull_bot = float(low[ob_i])
            act_bull_i = ob_i
        if bear_disp[i] and last_bull_idx >= 0:
            ob_i = last_bull_idx
            act_bear_top = float(high[ob_i])
            act_bear_bot = float(low[ob_i])
            act_bear_i = ob_i

        if act_bull_top is not None:
            bull_ob_top[i] = act_bull_top
            bull_ob_bottom[i] = act_bull_bot
            bull_ob_idx[i] = act_bull_i
            bull_ob_mit[i] = False
        else:
            bull_ob_mit[i] = True
        if act_bear_top is not None:
            bear_ob_top[i] = act_bear_top
            bear_ob_bottom[i] = act_bear_bot
            bear_ob_idx[i] = act_bear_i
            bear_ob_mit[i] = False
        else:
            bear_ob_mit[i] = True

        # Track most recent opposing candle (direction 0 / doji doesn't
        # advance either pointer so we always have a valid OB source).
        if direction[i] == -1:
            last_bear_idx = i
        elif direction[i] == 1:
            last_bull_idx = i

    df["bull_ob_top"] = bull_ob_top
    df["bull_ob_bottom"] = bull_ob_bottom
    df["bull_ob_idx"] = bull_ob_idx
    df["bull_ob_mitigated"] = bull_ob_mit
    df["bear_ob_top"] = bear_ob_top
    df["bear_ob_bottom"] = bear_ob_bottom
    df["bear_ob_idx"] = bear_ob_idx
    df["bear_ob_mitigated"] = bear_ob_mit


# --------------------------------------------------------------------------- #
# Fair Value Gaps (3-candle imbalance)
# --------------------------------------------------------------------------- #
def _add_fvgs(df: pd.DataFrame) -> None:
    """Bull FVG = low[i] > high[i-2] (gap up between t=i-2 and t=i, middle i-1).
    Bear FVG = high[i] < low[i-2].
    Confirmed at bar i close.  Mitigated when a subsequent WICK trades
    through the gap boundary (low < bull_bot or high > bear_top).
    """
    n = len(df)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    bull_fvg_top = np.full(n, np.nan, dtype=np.float64)
    bull_fvg_bottom = np.full(n, np.nan, dtype=np.float64)
    bull_fvg_idx = np.full(n, -1, dtype=np.int64)
    bull_fvg_mit = np.zeros(n, dtype=bool)
    bear_fvg_top = np.full(n, np.nan, dtype=np.float64)
    bear_fvg_bottom = np.full(n, np.nan, dtype=np.float64)
    bear_fvg_idx = np.full(n, -1, dtype=np.int64)
    bear_fvg_mit = np.zeros(n, dtype=bool)

    act_bull_top: float | None = None
    act_bull_bot: float | None = None
    act_bull_mid: int = -1
    act_bear_top: float | None = None
    act_bear_bot: float | None = None
    act_bear_mid: int = -1

    for i in range(n):
        # Wick-based mitigation (same rule as OBs: low < bull_bot fills the gap).
        if act_bull_top is not None and low[i] < act_bull_bot:
            act_bull_top = act_bull_bot = None
            act_bull_mid = -1
        if act_bear_bot is not None and high[i] > act_bear_top:
            act_bear_top = act_bear_bot = None
            act_bear_mid = -1

        if i >= 2:
            t = i - 2
            if low[i] > high[t]:
                # Bull FVG gap UP: zone is [high[t], low[i]] (bot=high[t], top=low[i]).
                act_bull_top = float(low[i])
                act_bull_bot = float(high[t])
                act_bull_mid = t + 1
            if high[i] < low[t]:
                # Bear FVG gap DOWN: zone is [high[i], low[t]] (bot=high[i], top=low[t]).
                act_bear_top = float(low[t])
                act_bear_bot = float(high[i])
                act_bear_mid = t + 1

        if act_bull_top is not None:
            bull_fvg_top[i] = act_bull_top
            bull_fvg_bottom[i] = act_bull_bot
            bull_fvg_idx[i] = act_bull_mid
            bull_fvg_mit[i] = False
        else:
            bull_fvg_mit[i] = True
        if act_bear_top is not None:
            bear_fvg_top[i] = act_bear_top
            bear_fvg_bottom[i] = act_bear_bot
            bear_fvg_idx[i] = act_bear_mid
            bear_fvg_mit[i] = False
        else:
            bear_fvg_mit[i] = True

    df["bull_fvg_top"] = bull_fvg_top
    df["bull_fvg_bottom"] = bull_fvg_bottom
    df["bull_fvg_idx"] = bull_fvg_idx
    df["bull_fvg_mitigated"] = bull_fvg_mit
    df["bear_fvg_top"] = bear_fvg_top
    df["bear_fvg_bottom"] = bear_fvg_bottom
    df["bear_fvg_idx"] = bear_fvg_idx
    df["bear_fvg_mitigated"] = bear_fvg_mit


# --------------------------------------------------------------------------- #
# Liquidity sweeps (stop-runs)
#
# A bull LIQUIDITY SWEEP occurs when the wick takes out a recent minor swing
# LOW (sellside liquidity hunted below stops) then closes BACK ABOVE that
# swing low -- rejection + reversal displacement. That wick low is the sweep
# extreme; SL goes beyond it. The sweep remains "live" as a bullish scalp
# setup until price closes below it (sweep fails = continuation).
#
# Bear sweep is symmetric: wick takes recent minor swing HIGH, closes back
# below it -- buyside liquidity hunted before a drop.
#
# The sweep is *active for reversal* for up to `sweep_window` bars after the
# sweep candle close, AND while price hasn't closed back beyond the sweep
# extreme (invalidation).
# --------------------------------------------------------------------------- #
def _add_liquidity_sweeps(df: pd.DataFrame) -> None:
    n = len(df)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    min_sh = df["minor_swing_high"].ffill().to_numpy(dtype=np.float64)
    min_sl = df["minor_swing_low"].ffill().to_numpy(dtype=np.float64)

    bull_sweep = np.zeros(n, dtype=bool)
    bear_sweep = np.zeros(n, dtype=bool)
    bull_sweep_px = np.full(n, np.nan, dtype=np.float64)
    bear_sweep_px = np.full(n, np.nan, dtype=np.float64)

    # Track last swing high/low that's been "printed" (available to be swept)
    # We need at least 1 bar after the swing pivot for it to exist.
    # wick_penetration_pts is the number of points past the swing we need
    # to count as a sweep (not just a tap). For XAUUSD this is a small buffer.
    wick_buffer_pts = 0.05  # ~5 cents on XAUUSD -- any wick past the swing counts

    last_bull_extreme: float | None = None  # last active bull sweep low
    last_bear_extreme: float | None = None  # last active bear sweep high

    for i in range(n):
        h = float(high[i])
        l = float(low[i])
        c = float(close[i])

        # Carry forward previous sweep levels (active until invalidated).
        if last_bull_extreme is not None:
            # Invalidation: price CLOSES below the sweep low => sweep fails,
            # it was a continuation not a reversal.
            if c < last_bull_extreme - wick_buffer_pts:
                last_bull_extreme = None
            else:
                bull_sweep_px[i] = last_bull_extreme
        if last_bear_extreme is not None:
            if c > last_bear_extreme + wick_buffer_pts:
                last_bear_extreme = None
            else:
                bear_sweep_px[i] = last_bear_extreme

        # Detect NEW sweeps on this bar.
        # Bull sweep: low[i] takes out last minor swing low, close[i] back above it
        ref_sl = float(min_sl[i]) if np.isfinite(min_sl[i]) else np.nan
        if np.isfinite(ref_sl) and l < ref_sl - wick_buffer_pts and c > ref_sl:
            # Wick took out the swing low and closed back above => bull sweep
            bull_sweep[i] = True
            last_bull_extreme = l
            bull_sweep_px[i] = l

        # Bear sweep: high[i] takes out last minor swing high, close[i] back below it
        ref_sh = float(min_sh[i]) if np.isfinite(min_sh[i]) else np.nan
        if np.isfinite(ref_sh) and h > ref_sh + wick_buffer_pts and c < ref_sh:
            bear_sweep[i] = True
            last_bear_extreme = h
            bear_sweep_px[i] = h

    df["bull_liq_sweep"] = bull_sweep
    df["bear_liq_sweep"] = bear_sweep
    df["bull_sweep_px"] = bull_sweep_px
    df["bear_sweep_px"] = bear_sweep_px


# --------------------------------------------------------------------------- #
# Premium / Discount within current major swing range
# --------------------------------------------------------------------------- #
def _add_premium_discount(df: pd.DataFrame) -> None:
    hi = df["major_swing_high"]
    lo = df["major_swing_low"]
    rng = (hi - lo).replace(0.0, np.nan)
    pct = ((df["close"] - lo) / rng).clip(lower=0.0, upper=1.0)
    df["range_high"] = hi.astype(np.float64)
    df["range_low"] = lo.astype(np.float64)
    df["price_in_range_pct"] = pct.fillna(0.5).astype(np.float64)
    df["is_premium"] = (pct > 0.67).fillna(False).astype(bool)
    df["is_discount"] = (pct < 0.33).fillna(False).astype(bool)
    df["is_equilibrium"] = (~(df["is_premium"] | df["is_discount"])).astype(bool)


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #


def _add_zone_proximity(df: pd.DataFrame) -> None:
    """Compute dynamic proximity to nearest unmitigated OB, FVG, and sweep level.

    All distances normalized by ATR so they are scale-invariant.
    Lower values = closer to zone = higher probability setup.
    """
    close = df["close"].values
    atr = df.get("atr_14", pd.Series(1.0, index=df.index)).values
    atr = np.where(np.isnan(atr) | (atr <= 0), 1.0, atr)

    # --- OB proximity ---
    bull_ob_top = df.get("bull_ob_top", pd.Series(np.nan, index=df.index)).values
    bull_ob_bot = df.get("bull_ob_bottom", pd.Series(np.nan, index=df.index)).values
    bull_ob_mit = df.get("bull_ob_mitigated", pd.Series(True, index=df.index)).values
    bear_ob_top = df.get("bear_ob_top", pd.Series(np.nan, index=df.index)).values
    bear_ob_bot = df.get("bear_ob_bottom", pd.Series(np.nan, index=df.index)).values
    bear_ob_mit = df.get("bear_ob_mitigated", pd.Series(True, index=df.index)).values

    # Distance to nearest unmitigated OB center
    bull_ob_mid = np.where(
        ~np.isnan(bull_ob_top) & ~np.isnan(bull_ob_bot) & ~bull_ob_mit,
        (bull_ob_top + bull_ob_bot) / 2.0,
        np.nan,
    )
    bear_ob_mid = np.where(
        ~np.isnan(bear_ob_top) & ~np.isnan(bear_ob_bot) & ~bear_ob_mit,
        (bear_ob_top + bear_ob_bot) / 2.0,
        np.nan,
    )
    dist_bull_ob = np.abs(close - bull_ob_mid) / atr
    dist_bear_ob = np.abs(close - bear_ob_mid) / atr
    # Nearest OB (either direction)
    ob_proximity = np.nanmin(np.column_stack([dist_bull_ob, dist_bear_ob]), axis=1)
    ob_proximity = np.where(np.isnan(ob_proximity), 10.0, ob_proximity)  # 10 = far away
    ob_proximity = np.clip(ob_proximity, 0.0, 10.0)

    # --- FVG proximity ---
    bull_fvg_top = df.get("bull_fvg_top", pd.Series(np.nan, index=df.index)).values
    bull_fvg_bot = df.get("bull_fvg_bottom", pd.Series(np.nan, index=df.index)).values
    bull_fvg_mit = df.get("bull_fvg_mitigated", pd.Series(True, index=df.index)).values
    bear_fvg_top = df.get("bear_fvg_top", pd.Series(np.nan, index=df.index)).values
    bear_fvg_bot = df.get("bear_fvg_bottom", pd.Series(np.nan, index=df.index)).values
    bear_fvg_mit = df.get("bear_fvg_mitigated", pd.Series(True, index=df.index)).values

    bull_fvg_mid = np.where(
        ~np.isnan(bull_fvg_top) & ~np.isnan(bull_fvg_bot) & ~bull_fvg_mit,
        (bull_fvg_top + bull_fvg_bot) / 2.0,
        np.nan,
    )
    bear_fvg_mid = np.where(
        ~np.isnan(bear_fvg_top) & ~np.isnan(bear_fvg_bot) & ~bear_fvg_mit,
        (bear_fvg_top + bear_fvg_bot) / 2.0,
        np.nan,
    )
    dist_bull_fvg = np.abs(close - bull_fvg_mid) / atr
    dist_bear_fvg = np.abs(close - bear_fvg_mid) / atr
    fvg_proximity = np.nanmin(np.column_stack([dist_bull_fvg, dist_bear_fvg]), axis=1)
    fvg_proximity = np.where(np.isnan(fvg_proximity), 10.0, fvg_proximity)
    fvg_proximity = np.clip(fvg_proximity, 0.0, 10.0)

    # --- Sweep proximity ---
    bull_sweep_px = df.get("bull_sweep_px", pd.Series(np.nan, index=df.index)).values
    bear_sweep_px = df.get("bear_sweep_px", pd.Series(np.nan, index=df.index)).values
    dist_bull_sweep = np.abs(close - bull_sweep_px) / atr
    dist_bear_sweep = np.abs(close - bear_sweep_px) / atr
    sweep_proximity = np.nanmin(np.column_stack([dist_bull_sweep, dist_bear_sweep]), axis=1)
    sweep_proximity = np.where(np.isnan(sweep_proximity), 10.0, sweep_proximity)
    sweep_proximity = np.clip(sweep_proximity, 0.0, 10.0)

    df["ob_proximity"] = ob_proximity
    df["fvg_proximity"] = fvg_proximity
    df["sweep_proximity"] = sweep_proximity



def _add_sr_zones(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    """Compute Support/Resistance zones from swing pivots.

    Uses the minor and major swing highs/lows to identify key S/R levels.
    - sr_support: nearest support level below current price (from swing lows)
    - sr_resistance: nearest resistance level above current price (from swing highs)
    - sr_support_dist: distance to support / ATR
    - sr_resistance_dist: distance to resistance / ATR
    - sr_count_support: number of support levels within 3 ATR (confluence)
    - sr_count_resistance: number of resistance levels within 3 ATR
    - at_support: bool — price is within 0.3 ATR of support
    - at_resistance: bool — price is within 0.3 ATR of resistance
    """
    close = df["close"].values
    n = len(df)
    atr = df.get("atr_14", pd.Series(1.0, index=df.index)).values
    atr = np.where(np.isnan(atr) | (atr <= 0), 1.0, atr)

    # Collect swing low/high prices as support/resistance
    sl_cols = [c for c in df.columns if "swing_low" in c and "idx" not in c and c.endswith("_price") or c == "swing_low"]
    sh_cols = [c for c in df.columns if "swing_high" in c and "idx" not in c and c.endswith("_price") or c == "swing_high"]

    # Also use generic swing columns if specific ones don't exist
    if not sl_cols:
        sl_cols = [c for c in df.columns if c in ("swing_low", "minor_swing_low", "major_swing_low")]
    if not sh_cols:
        sh_cols = [c for c in df.columns if c in ("swing_high", "minor_swing_high", "major_swing_high")]

    # Fallback: use OB/FVG edges as S/R levels too
    sr_support = np.full(n, np.nan)
    sr_resistance = np.full(n, np.nan)
    sr_support_dist = np.full(n, 10.0)
    sr_resistance_dist = np.full(n, 10.0)
    sr_count_support = np.zeros(n)
    sr_count_resistance = np.zeros(n)

    for i in range(n):
        supports = []
        resistances = []

        # Swing lows = support
        for col in sl_cols:
            if col in df.columns:
                val = df[col].iloc[i]
                if not np.isnan(val) and val > 0:
                    supports.append(val)

        # Swing highs = resistance
        for col in sh_cols:
            if col in df.columns:
                val = df[col].iloc[i]
                if not np.isnan(val) and val > 0:
                    resistances.append(val)

        # OB edges as S/R
        for col in ("bull_ob_bottom", "bear_ob_bottom"):
            val = df.get(col, pd.Series(np.nan, index=df.index)).iloc[i]
            if not np.isnan(val) and val > 0:
                supports.append(val)
        for col in ("bull_ob_top", "bear_ob_top"):
            val = df.get(col, pd.Series(np.nan, index=df.index)).iloc[i]
            if not np.isnan(val) and val > 0:
                resistances.append(val)

        # Nearest support below price
        below = [s for s in supports if s < close[i]]
        above = [r for r in resistances if r > close[i]]

        if below:
            sr_support[i] = max(below)  # nearest = highest below
            sr_support_dist[i] = (close[i] - sr_support[i]) / atr[i]
            sr_count_support[i] = sum(1 for s in below if (close[i] - s) / atr[i] < 3.0)

        if above:
            sr_resistance[i] = min(above)  # nearest = lowest above
            sr_resistance_dist[i] = (sr_resistance[i] - close[i]) / atr[i]
            sr_count_resistance[i] = sum(1 for r in above if (r - close[i]) / atr[i] < 3.0)

    df["sr_support"] = sr_support
    df["sr_resistance"] = sr_resistance
    df["sr_support_dist"] = np.clip(sr_support_dist, 0.0, 10.0)
    df["sr_resistance_dist"] = np.clip(sr_resistance_dist, 0.0, 10.0)
    df["sr_support_count"] = sr_count_support
    df["sr_resistance_count"] = sr_count_resistance
    df["at_support"] = (sr_support_dist < 0.3).astype(bool)
    df["at_resistance"] = (sr_resistance_dist < 0.3).astype(bool)


def _add_supply_demand_zones(df: pd.DataFrame) -> None:
    """Identify Supply/Demand zones from OB + FVG + sweep confluence.

    A Demand zone = area where buyers stepped in strongly (bull OB + bull FVG + sweep)
    A Supply zone = area where sellers stepped in strongly (bear OB + bear FVG + sweep)

    Features:
    - in_demand_zone: bool — price is in an active demand zone
    - in_supply_zone: bool — price is in an active supply zone
    - demand_zone_strength: 0-3 score (how many confirmations: OB+FVG+sweep)
    - supply_zone_strength: 0-3 score
    - demand_zone_dist: distance to nearest demand zone center / ATR
    - supply_zone_dist: distance to nearest supply zone center / ATR
    """
    close = df["close"].values
    n = len(df)
    atr = df.get("atr_14", pd.Series(1.0, index=df.index)).values
    atr = np.where(np.isnan(atr) | (atr <= 0), 1.0, atr)

    # Demand zone: bull OB + bull FVG overlap + bull sweep nearby
    bull_ob_top = df.get("bull_ob_top", pd.Series(np.nan, index=df.index)).values
    bull_ob_bot = df.get("bull_ob_bottom", pd.Series(np.nan, index=df.index)).values
    bull_ob_mit = df.get("bull_ob_mitigated", pd.Series(True, index=df.index)).values
    bull_fvg_top = df.get("bull_fvg_top", pd.Series(np.nan, index=df.index)).values
    bull_fvg_bot = df.get("bull_fvg_bottom", pd.Series(np.nan, index=df.index)).values
    bull_fvg_mit = df.get("bull_fvg_mitigated", pd.Series(True, index=df.index)).values
    bull_sweep = df.get("bull_liq_sweep", pd.Series(False, index=df.index)).values

    bear_ob_top = df.get("bear_ob_top", pd.Series(np.nan, index=df.index)).values
    bear_ob_bot = df.get("bear_ob_bottom", pd.Series(np.nan, index=df.index)).values
    bear_ob_mit = df.get("bear_ob_mitigated", pd.Series(True, index=df.index)).values
    bear_fvg_top = df.get("bear_fvg_top", pd.Series(np.nan, index=df.index)).values
    bear_fvg_bot = df.get("bear_fvg_bottom", pd.Series(np.nan, index=df.index)).values
    bear_fvg_mit = df.get("bear_fvg_mitigated", pd.Series(True, index=df.index)).values
    bear_sweep = df.get("bear_liq_sweep", pd.Series(False, index=df.index)).values

    in_demand = np.zeros(n, dtype=bool)
    in_supply = np.zeros(n, dtype=bool)
    demand_strength = np.zeros(n, dtype=np.float64)
    supply_strength = np.zeros(n, dtype=np.float64)
    demand_dist = np.full(n, 10.0)
    supply_dist = np.full(n, 10.0)

    for i in range(n):
        price = close[i]

        # Demand zone check
        d_strength = 0
        d_top, d_bot = np.nan, np.nan
        if not bull_ob_mit[i] and not np.isnan(bull_ob_top[i]):
            d_strength += 1
            d_top, d_bot = bull_ob_top[i], bull_ob_bot[i]
        if not bull_fvg_mit[i] and not np.isnan(bull_fvg_top[i]):
            d_strength += 1
            if np.isnan(d_top):
                d_top, d_bot = bull_fvg_top[i], bull_fvg_bot[i]
            else:
                d_top = max(d_top, bull_fvg_top[i])
                d_bot = min(d_bot, bull_fvg_bot[i])
        if bull_sweep[i]:
            d_strength += 1

        demand_strength[i] = d_strength
        if d_strength > 0 and not np.isnan(d_top):
            if d_bot <= price <= d_top:
                in_demand[i] = True
            demand_dist[i] = abs(price - (d_top + d_bot) / 2.0) / atr[i]

        # Supply zone check
        s_strength = 0
        s_top, s_bot = np.nan, np.nan
        if not bear_ob_mit[i] and not np.isnan(bear_ob_top[i]):
            s_strength += 1
            s_top, s_bot = bear_ob_top[i], bear_ob_bot[i]
        if not bear_fvg_mit[i] and not np.isnan(bear_fvg_top[i]):
            s_strength += 1
            if np.isnan(s_top):
                s_top, s_bot = bear_fvg_top[i], bear_fvg_bot[i]
            else:
                s_top = max(s_top, bear_fvg_top[i])
                s_bot = min(s_bot, bear_fvg_bot[i])
        if bear_sweep[i]:
            s_strength += 1

        supply_strength[i] = s_strength
        if s_strength > 0 and not np.isnan(s_top):
            if s_bot <= price <= s_top:
                in_supply[i] = True
            supply_dist[i] = abs(price - (s_top + s_bot) / 2.0) / atr[i]

    df["in_demand_zone"] = in_demand
    df["in_supply_zone"] = in_supply
    df["demand_zone_strength"] = demand_strength
    df["supply_zone_strength"] = supply_strength
    df["demand_zone_dist"] = np.clip(demand_dist, 0.0, 10.0)
    df["supply_zone_dist"] = np.clip(supply_dist, 0.0, 10.0)


def _add_atr_regime(df: pd.DataFrame, cfg: FeatureConfig) -> None:
    """ATR percentile rank — is current volatility high or low relative to history?

    Features:
    - atr_pct_rank: 0-1 percentile rank of current ATR vs rolling 252-bar window
    - atr_expanding: bool — ATR > 1.2x its 20-bar SMA (volatility expanding)
    - atr_contracting: bool — ATR < 0.8x its 20-bar SMA (volatility contracting)
    """
    atr = df.get("atr_14", pd.Series(1.0, index=df.index)).values
    atr = np.where(np.isnan(atr) | (atr <= 0), 1.0, atr)

    atr_series = pd.Series(atr)
    atr_sma20 = atr_series.rolling(20, min_periods=1).mean()

    # Percentile rank over rolling window
    window = min(252, len(atr))
    pct_rank = atr_series.rolling(window, min_periods=1).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x), raw=False
    ).fillna(0.5)

    df["atr_pct_rank"] = pct_rank.values
    df["atr_expanding"] = (atr_series > 1.2 * atr_sma20).values.astype(bool)
    df["atr_contracting"] = (atr_series < 0.8 * atr_sma20).values.astype(bool)


def process_bars(
    df: pd.DataFrame,
    timeframe: str,
    cfg: FeatureConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    tick_dir: Path | None = None,
) -> pd.DataFrame:
    """Enrich a single-TF OHLCV dataframe with all Layer-2 features.

    Input ``df`` must have columns: time, open, high, low, close,
    tick_volume, spread, real_volume (raw mt5 bars schema). Must be
    sorted by time ascending.  ``timeframe`` is informational (used in
    progress messages); swing detection is TF-agnostic via the
    ATR-ZigZag default.
    """
    if progress:
        progress(f"    {timeframe}: preparing ...")
    cfg = cfg or DEFAULT_CONFIG
    out = df.copy().reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype(np.float64)
    out["tick_volume"] = pd.to_numeric(out["tick_volume"], errors="coerce").fillna(0).astype(np.float64)

    if progress:
        progress(f"    {timeframe}: sessions/candle math ...")
    _add_sessions(out)
    _add_basics(out, cfg)
    if progress:
        progress(f"    {timeframe}: volatility/EMAs ...")
    _add_volatility(out, cfg)
    _add_emas(out, cfg)
    _add_volume(out, cfg)
    if progress:
        progress(f"    {timeframe}: swings/structure ({cfg.swing_method}) ...")
    _add_structure(out, cfg)
    _add_displacement(out, cfg)
    if progress:
        progress(f"    {timeframe}: order blocks + FVGs ...")
    _add_order_blocks(out)
    _add_fvgs(out)
    _add_liquidity_sweeps(out)
    _add_premium_discount(out)
    # v1.0: ICT/SMC concepts
    _add_silver_bullet(out)
    _add_power_of_3(out)
    _add_judas_swing(out)
    ce_cols = _add_consequent_encroachment(out)
    out = pd.concat([out, ce_cols], axis=1)
    if progress:
        progress(f"    {timeframe}: S/R zones + S/D zones + ATR regime ...")
    _add_sr_zones(out, cfg)
    _add_supply_demand_zones(out)
    _add_atr_regime(out, cfg)

    if progress:
        progress(f"    {timeframe}: zone proximity ...")
    _add_zone_proximity(out)

    # Tick-derived features (M1 only — ticks don't exist for HTFs)
    if timeframe == "M1" and tick_dir is not None:
        if progress:
            progress(f"    {timeframe}: tick features ...")
        from .tick_features import compute_tick_features
        tick_cols = compute_tick_features(out, tick_dir)
        out = pd.concat([out, tick_cols], axis=1)
        if progress:
            n_nonzero = (tick_cols["tick_count"] > 0).sum()
            progress(f"    {timeframe}: tick features ({n_nonzero:,}/{len(out):,} bars with ticks)")

    if progress:
        progress(f"    {timeframe}: done ({len(out):,} rows, {len(out.columns)} cols)")
    return out


# --------------------------------------------------------------------------- #
# v1.0: ICT/SMC concepts — Silver Bullet, Power of 3, Judas Swing, CE
# --------------------------------------------------------------------------- #

def _add_silver_bullet(df: pd.DataFrame) -> None:
    """Add Silver Bullet time windows (ICT 2022 Mentorship).

    Silver Bullet windows are high-probability entry zones:
    - London Silver Bullet: 03:00-05:00 EST (08:00-10:00 UTC)
    - NY Silver Bullet: 10:00-12:00 EST (15:00-17:00 UTC)
    - AM Silver Bullet: 02:00-04:00 EST (07:00-09:00 UTC) — overlap

    During these windows, ICT says institutional algorithms drive price
    to sweep liquidity before the real move. Entries during these windows
    have higher probability of success.
    """
    ts = df["time"].dt
    h = ts.hour.astype(np.int32)
    m = ts.minute.astype(np.int32)
    hm = h * 60 + m

    # London Silver Bullet: 08:00-10:00 UTC
    df["silver_bullet_london"] = ((hm >= 480) & (hm < 600)).astype(bool)
    # NY Silver Bullet: 15:00-17:00 UTC
    df["silver_bullet_ny"] = ((hm >= 900) & (hm < 1020)).astype(bool)
    # Any Silver Bullet window
    df["in_silver_bullet"] = (df["silver_bullet_london"] | df["silver_bullet_ny"]).astype(bool)


def _add_power_of_3(df: pd.DataFrame) -> None:
    """Add Power of 3 (AMD) cycle detection per session.

    ICT's Power of 3: Accumulation → Manipulation → Distribution
    - Accumulation: price consolidates in a range (low volatility)
    - Manipulation: price sweeps one side of the range (liquidity grab)
    - Distribution: price reverses and moves to the other side (real move)

    We detect this by tracking the session range and identifying:
    - Asian range (consolidation)
    - London/NY sweep of Asian high/low (manipulation)
    - Reversal after sweep (distribution)
    """
    ts = df["time"].dt
    h = ts.hour.astype(np.int32)
    hm = h * 60 + ts.minute.astype(np.int32)

    # Track Asian range high/low (00:00-08:00 UTC)
    is_asian = hm < 480  # before London open
    is_london = (hm >= 480) & (hm < 780)  # 08:00-13:00 UTC
    is_ny = (hm >= 780) & (hm < 1020)  # 13:00-17:00 UTC

    # Rolling Asian range (using expanding window within each day)
    df["_asian_hi"] = np.where(is_asian, df["high"], np.nan)
    df["_asian_lo"] = np.where(is_asian, df["low"], np.nan)

    # Forward-fill Asian range to London/NY sessions
    df["_asian_hi"] = df["_asian_hi"].ffill()
    df["_asian_lo"] = df["_asian_lo"].ffill()

    # Detect sweep of Asian range during London/NY
    asian_hi = df["_asian_hi"].values
    asian_lo = df["_asian_lo"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Sweep high: price trades above Asian high then closes below
    sweep_asian_high = (high > asian_hi) & (close < asian_hi) & (is_london | is_ny)
    # Sweep low: price trades below Asian low then closes above
    sweep_asian_low = (low < asian_lo) & (close > asian_lo) & (is_london | is_ny)

    df["pow3_sweep_high"] = sweep_asian_high.astype(bool)
    df["pow3_sweep_low"] = sweep_asian_low.astype(bool)
    df["in_pow3_manipulation"] = (sweep_asian_high | sweep_asian_low).astype(bool)

    # Clean up temp columns
    df.drop(columns=["_asian_hi", "_asian_lo"], inplace=True, errors="ignore")


def _add_judas_swing(df: pd.DataFrame) -> None:
    """Add Judas Swing detection.

    ICT's Judas Swing: a fake move at session open that traps traders
    before the real move in the opposite direction.

    Detection:
    - At London open (08:00 UTC): if price moves UP in first 30min then
      reverses DOWN = bearish Judas (trap longs)
    - At London open: if price moves DOWN in first 30min then reverses
      UP = bullish Judas (trap shorts)
    - Same pattern at NY open (13:30 UTC)

    We detect this by comparing the first 30min direction with the
    subsequent 30min direction.
    """
    ts = df["time"].dt
    h = ts.hour.astype(np.int32)
    m = ts.minute.astype(np.int32)
    hm = h * 60 + m

    close = df["close"].values
    open_ = df["open"].values

    # London Judas: first 30min (08:00-08:30) vs next 30min (08:30-09:00)
    in_lon_first30 = (hm >= 480) & (hm < 510)
    in_lon_next30 = (hm >= 510) & (hm < 540)

    # NY Judas: first 30min (13:30-14:00) vs next 30min (14:00-14:30)
    in_ny_first30 = (hm >= 810) & (hm < 840)
    in_ny_next30 = (hm >= 840) & (hm < 870)

    # Direction of first 30min candle
    first30_dir = np.where(close > open_, 1, np.where(close < open_, -1, 0))

    # Judas = first30 direction opposite to subsequent move
    # We flag bars in the "next 30min" if they reverse the first30 move
    judas_bull = np.zeros(len(df), dtype=bool)
    judas_bear = np.zeros(len(df), dtype=bool)

    # Simple heuristic: if first30 was bearish and next30 is bullish = bullish Judas
    # This is a simplified version — full implementation would track the actual
    # first30 range and detect the sweep+reversal
    for i in range(1, len(df)):
        if in_lon_next30[i] or in_ny_next30[i]:
            # Look back to find the first30 candle
            for j in range(i-1, max(0, i-10), -1):
                if in_lon_first30[j] or in_ny_first30[j]:
                    if first30_dir[j] == -1 and close[i] > open_[i]:
                        judas_bull[i] = True  # Bullish Judas (trapped shorts)
                    elif first30_dir[j] == 1 and close[i] < open_[i]:
                        judas_bear[i] = True  # Bearish Judas (trapped longs)
                    break

    df["judas_bull"] = judas_bull
    df["judas_bear"] = judas_bear
    df["in_judas_swing"] = (judas_bull | judas_bear).astype(bool)


def _add_consequent_encroachment(df: pd.DataFrame) -> pd.DataFrame:
    """Add Consequent Encroachment (CE) — 50% midpoint of FVGs.

    ICT says the 50% level of a Fair Value Gap is a high-probability
    reaction zone. When price returns to the CE, it often bounces.

    For each active FVG:
    - CE = (top + bottom) / 2
    - Track distance from current price to nearest CE
    """
    close = df["close"].values

    # Bull FVG CE
    bull_fvg_top = df.get("bull_fvg_top", pd.Series(np.nan, index=df.index)).values
    bull_fvg_bot = df.get("bull_fvg_bottom", pd.Series(np.nan, index=df.index)).values
    bull_ce = np.where(
        ~np.isnan(bull_fvg_top) & ~np.isnan(bull_fvg_bot),
        (bull_fvg_top + bull_fvg_bot) / 2.0,
        np.nan,
    )

    # Bear FVG CE
    bear_fvg_top = df.get("bear_fvg_top", pd.Series(np.nan, index=df.index)).values
    bear_fvg_bot = df.get("bear_fvg_bottom", pd.Series(np.nan, index=df.index)).values
    bear_ce = np.where(
        ~np.isnan(bear_fvg_top) & ~np.isnan(bear_fvg_bot),
        (bear_fvg_top + bear_fvg_bot) / 2.0,
        np.nan,
    )

    # Distance to nearest CE (normalized by ATR)
    atr = df.get("atr_14", pd.Series(1.0, index=df.index)).values
    atr = np.where(np.isnan(atr) | (atr <= 0), 1.0, atr)
    dist_bull_ce = np.abs(close - bull_ce) / atr
    dist_bear_ce = np.abs(close - bear_ce) / atr

    # Batch all new columns at once to avoid DataFrame fragmentation
    new_cols = pd.DataFrame({
        "bull_ce": bull_ce,
        "bear_ce": bear_ce,
        "dist_to_bull_ce": dist_bull_ce,
        "dist_to_bear_ce": dist_bear_ce,
        "at_bull_ce": (dist_bull_ce < 0.3).astype(bool),
        "at_bear_ce": (dist_bear_ce < 0.3).astype(bool),
    }, index=df.index)
    return new_cols
