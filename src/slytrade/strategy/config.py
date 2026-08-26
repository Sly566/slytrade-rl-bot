"""Layer 4 strategy config — ICT/SMC scalper.

All tunables here; the signal/backtest engines consume them. Defaults are
calibrated for XAUUSD M1 but the code is asset-class agnostic (dynamic sizing
per `symbol_info` lives in the backtest/OMS layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Setup-grade sizing by confluence (A+ / A / B / C)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SetupGrades:
    """Risk-per-trade as fraction of account equity."""
    a_plus: float = 0.0100   # 1.00% — full structural confluence + premium zone
    a:      float = 0.0075   # 0.75% — strong confluence, minor missing piece
    b:      float = 0.0050   # 0.50% — standard setup
    c:      float = 0.0025   # 0.25% — lowest grade; counter-trend retests only

    def risk_fraction(self, grade: str) -> float:
        return getattr(self, grade.lower().replace('+', '_plus'))


# --------------------------------------------------------------------------- #
# Exit plan (laddered, applied the same way for long and short)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExitPlan:
    # TP1: take first profit, close `tp1_pct` of size, move SL to breakeven.
    # 0.85R is the Layer-5 scalp sweet-spot on XAUUSD M1 (25-month sweep, n=129, PF=2.00):
    # far enough that M5 displacement legs tend to extend, close enough that
    # win-rate stays ~65% with costs.  One-shot (tp1_pct=1.0) beats laddered
    # exits on PF after 25 months of data.
    tp1_r:       float = 0.85
    tp1_pct:     float = 1.00   # 100% off at TP1 for the scalp profile (partial exits + trailing optional)
    # TP2: take second profit, close `tp2_pct`, trail SL to TP1.
    # Set tp2_pct=0 to disable (pure one-shot scalp).
    tp2_r:       float = 1.5    # secondary target if tp1_pct < 1
    tp2_pct:     float = 0.00   # 0% off by default — one-shot scalp
    # Runner (remaining after TP2): trail stop by trail_atr_mult × ATR OR exit on M5 CHoCH against
    runner_trail_atr_mult: float = 0.5
    runner_stop_tf: str = "M5"          # CHoCH on this TF exits the runner
    # Emergency / time stops
    emergency_choch_tf: str = "M15"     # CHoCH on this TF => emergency exit full
    time_stop_bars:      int = 240      # M1 bars = 4 hours; if BE not hit -> close
    time_stop_min_r:     float = 0.0    # if price is within ±this R of entry, flat
    ob_invalidation_buffer: float = 0.05  # buffer past OB/FVG edge, in ATR multiples
                                         # (tightened from 0.1 to 0.05 ATR after
                                         # battle-testing — 0.1 inflated risk ~10%
                                         # which was eating the scalp edge)


# --------------------------------------------------------------------------- #
# Killzone / session filter
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionFilter:
    """Which killzones/sessions to accept entries in.

    Battle-tested (Layer 5 backtest, 2024-08 → 2026-08): London + NY killzones
    (including their 30-min open windows) are profitable; Asian C-grade range
    retests and OFF-hours are net negative after costs and are BLOCKED.
    """
    trade_london_kz:     bool = True
    trade_ny_kz:         bool = True
    trade_asian_kz:      bool = False   # Asia is chop — skip after backtest
    trade_london_open30: bool = True
    trade_ny_open30:     bool = True
    trade_asian_range_retest: bool = False  # was True; PF=0.80 -> disabled
    block_off_hours:     bool = True    # reject OFF-session entirely


# --------------------------------------------------------------------------- #
# Structural confluence thresholds (which HTFs must agree)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConfluenceConfig:
    """Which HTFs must agree to reach each grade.

    Bias agreement means major_bias matches trade direction. A CHoCH against
    on a higher TF is an INSTANT KILL (emergency exit or blocks entry).
    """
    # Must have BOS/CHoCH on trigger TF in direction (M5)
    require_trigger_bos_or_choch: bool = True
    # Order-blocks: which TFs do we accept OBs from (sorted highest-conviction first)
    ob_tfs: tuple[str, ...] = ("H1", "M15", "M5")
    # Premium/discount filter: M1 price must be in this zone of the range TF
    pd_range_tf: str = "M15"
    pd_zone_min_pct: float = 0.10   # entries below 10% from edge count as "in zone"
    pd_zone_max_pct: float = 0.40   # entries above 40% from edge are rejected
    # HTF bias alignment requirements per grade
    #   "a_plus":  all of D1, H4, H1 agree with direction
    #   "a":       H4 + H1 agree (D1 neutral ok)
    #   "b":       H1 + M15 agree
    #   "c":       M15 agrees (counter-H4 allowed only on strong trigger)
    a_plus_required_tfs: tuple[str, ...] = ("D1", "H4", "H1")
    a_required_tfs:      tuple[str, ...] = ("H4", "H1")
    b_required_tfs:      tuple[str, ...] = ("H1", "M15")
    c_required_tfs:      tuple[str, ...] = ("M15",)
    # Kill-zones count as confluence: two killzones overlap => grade bump
    killzone_confluence_bonus: bool = True
    # ATR filter: skip if ATR14 is too small (dead market) or spiking.
    min_atr_pct: float = 0.0004  # ~$1.0 ATR floor at $2500 XAU (was 0.0002 = $0.5)
    max_atr_pct: float = 0.02    # ≈$50 ATR FOMC/news spike (unchanged)
    # Risk-in-ATR floor: require the stop to be at least this many ATRs wide.
    # Layer 5 forensics (25-month M1 sweep, 726,936 bars): stops tighter than
    # ~2 ATR get hunted by M1 noise — PF collapses from 2.00 (>=2 ATR) to 1.26
    # at the old 1.2 floor.  2.0 ATR minimum is now the default.
    min_risk_atr: float = 2.0
    max_risk_atr: float = 8.0
    # Only accept OBs on these TFs (H1 OBs had PF=0.84 and are filtered out).
    # FVGs are also disabled at default (only OBs passed the OOS test).
    accept_ob_tfs: tuple[str, ...] = ("M15", "M5")
    accept_zone_kinds: tuple[str, ...] = ("OB",)  # "OB","FVG"
    # Only accept these grades (C-grade is net negative; blocked by default).
    accept_grades: tuple[str, ...] = ("A+", "A", "B")
    # Directional toggle — Layer 5 battle-tested: LONGS carry ~94% of the edge
    # (PF 2.15 vs shorts PF 1.01 on 25-month M1 sweep).  SHORTS disabled by default;
    # flip accept_shorts=True only if you've revalidated on fresh data.
    accept_longs: bool = True
    accept_shorts: bool = False


@dataclass(frozen=True)
class StrategyConfig:
    grades:     SetupGrades     = field(default_factory=SetupGrades)
    exits:      ExitPlan        = field(default_factory=ExitPlan)
    sessions:   SessionFilter   = field(default_factory=SessionFilter)
    confluence: ConfluenceConfig = field(default_factory=ConfluenceConfig)
    # Execution TFs
    execution_tf: str = "M1"
    trigger_tf:   str = "M5"   # displacement/BOS trigger TF
    entry_tf:     str = "M1"   # precision entry (FVG/OB retest)
