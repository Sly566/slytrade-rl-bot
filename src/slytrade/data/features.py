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

from dataclasses import dataclass
from typing import Callable

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
        - mitigated when a subsequent close < OB bottom
    """
    n = len(df)
    direction = df["direction"].to_numpy()
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
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
        if act_bull_top is not None and close[i] < act_bull_bot:
            act_bull_top = act_bull_bot = None
            act_bull_i = -1
        if act_bear_bot is not None and close[i] > act_bear_top:
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
    Confirmed at bar i close.  Mitigated when a subsequent close fills
    back through the gap boundary.
    """
    n = len(df)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

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
        if act_bull_top is not None and close[i] < act_bull_bot:
            act_bull_top = act_bull_bot = None
            act_bull_mid = -1
        if act_bear_bot is not None and close[i] > act_bear_top:
            act_bear_top = act_bear_bot = None
            act_bear_mid = -1

        if i >= 2:
            t = i - 2
            if low[i] > high[t]:
                act_bull_top = float(low[i])
                act_bull_bot = float(high[t])
                act_bull_mid = t + 1
            if high[i] < low[t]:
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
def process_bars(
    df: pd.DataFrame,
    timeframe: str,
    cfg: FeatureConfig | None = None,
    *,
    progress: Callable[[str], None] | None = None,
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
    _add_premium_discount(out)
    if progress:
        progress(f"    {timeframe}: done ({len(out):,} rows, {len(out.columns)} cols)")
    return out
