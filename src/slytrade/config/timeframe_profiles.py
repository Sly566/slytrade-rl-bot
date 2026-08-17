"""Validated per-timeframe strategy profiles.

Each timeframe has different cost economics (spread vs ATR) and a different
edge horizon, so the entry/exit structure MUST scale with it. These profiles
are the data-validated defaults, measured on 25 months of real Exness XAUUSD
with an honest chronological split (config picked on the first year, validated
on the held-out second year), net of commission + slippage:

    H1  : +17.7% / +9.1%  out-of-sample, PF 1.66/1.32, max DD 4.2%  (champion)
    M15 : +12.7% / +25.7% out-of-sample, PF 1.13/1.24, max DD 7.7%  (scalping)
    M5  : negative net (overtrading + ~7.6% cost/R)
    M1  : structurally unprofitable (~17.6% cost/R vs ~5% gross edge at scalp horizon)

Round-trip cost per unit of risk (audited on real data: spread + $3.5/lot/side
commission + 5 points slippage, risk-budgeted lots for a 1xATR stop):
    M1 17.6% · M5 7.6% · M15 4.3% · H1 2.1% · H4 1.1%

``configs/risk.yaml`` remains the manual override layer for the timeframe-
insensitive knobs (gates, SMC weights, trailing/partial/breakeven rules).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeProfile:
    """Entry/exit structure for one decision timeframe."""

    min_score: int
    cooldown_bars: int
    stop_loss_atr: float
    take_profit_atr: float
    max_bars_in_trade: int | None
    htf_trend_timeframe: str | None
    note: str


TIMEFRAME_PROFILES: dict[str, TimeframeProfile] = {
    "H1": TimeframeProfile(
        min_score=4, cooldown_bars=20, stop_loss_atr=1.0, take_profit_atr=2.0,
        max_bars_in_trade=None, htf_trend_timeframe="h4",
        note="+17.7% 1stY / +9.1% 2ndY OOS, PF 1.66/1.32, max DD 4.2% — the champion",
    ),
    "M15": TimeframeProfile(
        min_score=3, cooldown_bars=10, stop_loss_atr=1.0, take_profit_atr=3.0,
        max_bars_in_trade=60, htf_trend_timeframe="h4",
        note="+12.7% 1stY / +25.7% 2ndY OOS, PF 1.13/1.24 — the profitable scalping timeframe",
    ),
    "M5": TimeframeProfile(
        min_score=3, cooldown_bars=20, stop_loss_atr=1.0, take_profit_atr=2.0,
        max_bars_in_trade=None, htf_trend_timeframe="h4",
        note="negative net of costs (overtrading + ~7.6% cost/R) — not recommended",
    ),
    "M1": TimeframeProfile(
        min_score=4, cooldown_bars=20, stop_loss_atr=1.0, take_profit_atr=2.0,
        max_bars_in_trade=None, htf_trend_timeframe="h4",
        note="structurally unprofitable: ~17.6% cost/R vs ~5% gross edge at the scalp horizon",
    ),
    # Reasonable defaults for the remaining bar timeframes (not re-validated).
    "M30": TimeframeProfile(
        min_score=4, cooldown_bars=15, stop_loss_atr=1.0, take_profit_atr=2.0,
        max_bars_in_trade=None, htf_trend_timeframe="h4", note="defaults (unvalidated)",
    ),
    "H4": TimeframeProfile(
        min_score=4, cooldown_bars=5, stop_loss_atr=1.0, take_profit_atr=2.0,
        max_bars_in_trade=None, htf_trend_timeframe="d1", note="defaults (unvalidated)",
    ),
    "D1": TimeframeProfile(
        min_score=4, cooldown_bars=1, stop_loss_atr=1.0, take_profit_atr=2.0,
        max_bars_in_trade=None, htf_trend_timeframe=None, note="defaults (unvalidated)",
    ),
    "W1": TimeframeProfile(
        min_score=4, cooldown_bars=1, stop_loss_atr=1.0, take_profit_atr=2.0,
        max_bars_in_trade=None, htf_trend_timeframe=None, note="defaults (unvalidated)",
    ),
}

DEFAULT_PROFILE = TIMEFRAME_PROFILES["H1"]


def profile_for(timeframe: str | None) -> TimeframeProfile:
    """Return the validated profile for a timeframe (fallback: the H1 champion)."""
    if not timeframe:
        return DEFAULT_PROFILE
    return TIMEFRAME_PROFILES.get(timeframe.upper(), DEFAULT_PROFILE)
