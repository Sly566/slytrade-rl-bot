"""Layer 4 — ICT/SMC scalper signal engine.

Produces strictly-causal entry signals from the M1-aligned frame. Each signal
carries: direction, entry price, stop loss, take-profit ladder, setup grade,
confluence list, and debug metadata (which HTF bars produced each condition).

The engine is PURE: no lookahead, no position state, no broker interaction.
It just walks rows and emits signal dicts. Layer 5 (backtest) and Layer 6
(paper/live) consume these signals identically.

Hard-gated checklist philosophy (per Sly's spec):
  * Each gate is PASS/FAIL — no weighted scoring turns a FAIL into a PASS.
  * Confluence depth decides SIZE/GRADE (A+/A/B/C) but never bypasses a gate.
  * Ranging markets: displacement-only, no mean-reversion; wait for a break.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

from .config import ConfluenceConfig, StrategyConfig


# --------------------------------------------------------------------------- #
# Signal dataclass
# --------------------------------------------------------------------------- #
@dataclass
class Signal:
    time: pd.Timestamp                   # M1 bar OPEN time (decision moment)
    direction: int                       # +1 long, -1 short
    entry: float                         # intended entry price (limit or market)
    stop: float                          # invalidation price (beyond OB/FVG edge)
    tp1: float                           # 1R target
    tp2: float                           # 2R target
    tp_runner: float                     # HTF swing target (major HTF swing)
    risk_per_unit: float                 # |entry - stop| (USD per unit for XAUUSD)
    grade: str                           # "A+", "A", "B", "C"
    risk_pct: float                      # fraction of equity (from SetupGrades)
    setup_kind: str = "RETEST_OB"        # RETEST_OB | RETEST_FVG | LIQ_SWEEP | BOS_CONT
    confluence: list[str] = field(default_factory=list)
    fails: list[str] = field(default_factory=list)
    # Debug
    trigger_tf: str = "M5"
    ob_tf: str | None = None
    ob_top: float | None = None
    ob_bottom: float | None = None
    fvg_top: float | None = None
    fvg_bottom: float | None = None
    swing_target_tf: str | None = None
    swing_target_price: float | None = None
    atr_at_entry: float = 0.0
    htf_bias_summary: dict[str, int] = field(default_factory=dict)
    session: str = ""
    killzone: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d['time'] = self.time.isoformat()
        return d

    @property
    def r_multiple_tp1(self) -> float:
        return abs(self.tp1 - self.entry) / self.risk_per_unit if self.risk_per_unit else 0.0

    @property
    def r_multiple_tp2(self) -> float:
        return abs(self.tp2 - self.entry) / self.risk_per_unit if self.risk_per_unit else 0.0


# --------------------------------------------------------------------------- #
# Per-row condition evaluation helpers
# --------------------------------------------------------------------------- #


def _price_within_ob_bull(row, tf: str, close_col: str = 'close') -> tuple[float, float] | None:
    """If current M1 close is inside the bullish OB on `tf`, return (ob_top, ob_bottom)."""
    top = row.get(f'{tf}_bull_ob_top', np.nan)
    bot = row.get(f'{tf}_bull_ob_bottom', np.nan)
    mit = row.get(f'{tf}_bull_ob_mitigated', True)
    c = row[close_col]
    if pd.isna(top) or pd.isna(bot) or bool(mit):
        return None
    # "Inside OB" = close between bottom and top (OB is the last bearish
    # candle before bull displacement; we want price to come back to it)
    if bot <= c <= top:
        return (float(top), float(bot))
    return None


def _price_within_ob_bear(row, tf: str, close_col: str = 'close') -> tuple[float, float] | None:
    top = row.get(f'{tf}_bear_ob_top', np.nan)
    bot = row.get(f'{tf}_bear_ob_bottom', np.nan)
    mit = row.get(f'{tf}_bear_ob_mitigated', True)
    c = row[close_col]
    if pd.isna(top) or pd.isna(bot) or bool(mit):
        return None
    if bot <= c <= top:
        return (float(top), float(bot))
    return None


def _price_touching_fvg_bull(row, tf: str, close_col: str = 'close') -> tuple[float, float] | None:
    """Bull FVG: low[i] > high[i-2]; gap is (high[i-2], low[i]). Price returning
    to the gap = FVG fill area."""
    top = row.get(f'{tf}_bull_fvg_top', np.nan)
    bot = row.get(f'{tf}_bull_fvg_bottom', np.nan)
    mit = row.get(f'{tf}_bull_fvg_mitigated', True)
    c = row[close_col]
    if pd.isna(top) or pd.isna(bot) or bool(mit):
        return None
    if bot <= c <= top:
        return (float(top), float(bot))
    return None


def _price_touching_fvg_bear(row, tf: str, close_col: str = 'close') -> tuple[float, float] | None:
    top = row.get(f'{tf}_bear_fvg_top', np.nan)
    bot = row.get(f'{tf}_bear_fvg_bottom', np.nan)
    mit = row.get(f'{tf}_bear_fvg_mitigated', True)
    c = row[close_col]
    if pd.isna(top) or pd.isna(bot) or bool(mit):
        return None
    if bot <= c <= top:
        return (float(top), float(bot))
    return None


# --------------------------------------------------------------------------- #
# Grading: count how many required-TF biases agree and assign grade
# --------------------------------------------------------------------------- #

def _grade(direction: int,
           row: pd.Series,
           cfg: ConfluenceConfig,
           bonus_killzone: bool) -> tuple[str, list[str]]:
    """Return (grade, list_of_confluence_tags). 'fail' if no grade met.

    Tags are only added for tiers that *pass*, so we don't pollute the list
    with partial-tier fragments (which caused duplicate TF tags when a higher
    tier checked a TF that a lower tier also checked).
    """
    tags: list[str] = []

    def _all_agree(tfs: tuple[str, ...]) -> tuple[bool, list[str]]:
        """Return (all_ok, [tag_for_each_agreeing_tf])."""
        ok_tags: list[str] = []
        for tf in tfs:
            b = row.get(f'{tf}_major_bias', 0)
            if pd.isna(b) or int(b) != direction:
                return (False, [])
            ok_tags.append(f'{tf}_bias_aligned')
        return (True, ok_tags)

    grade: str | None = None
    extra: list[str] = []

    # A+ requires D1+H4+H1 all aligned AND price in pd_zone of pd_range_tf
    ok, tier_tags = _all_agree(cfg.a_plus_required_tfs)
    if ok:
        pd_tf = cfg.pd_range_tf
        pct = row.get(f'{pd_tf}_price_in_range_pct', np.nan)
        if direction == 1 and pd.notna(pct) and pct <= cfg.pd_zone_max_pct:
            extra.append(f'{pd_tf}_discount_long')
            grade = 'A+'
        elif direction == -1 and pd.notna(pct) and pct >= (1.0 - cfg.pd_zone_max_pct):
            extra.append(f'{pd_tf}_premium_short')
            grade = 'A+'
        else:
            extra.append('htf_align_a_plus_but_not_in_zone')
            grade = 'A'
        tags.extend(tier_tags)

    if grade is None:
        ok, tier_tags = _all_agree(cfg.a_required_tfs)
        if ok:
            grade = 'A'
            tags.extend(tier_tags)

    if grade is None:
        ok, tier_tags = _all_agree(cfg.b_required_tfs)
        if ok:
            grade = 'B'
            tags.extend(tier_tags)

    if grade is None:
        ok, tier_tags = _all_agree(cfg.c_required_tfs)
        if ok:
            grade = 'C'
            tags.extend(tier_tags)

    if grade is None:
        return ('fail', tags)

    tags.extend(extra)
    if bonus_killzone:
        tags.append('killzone_overlap')
    return (grade, tags)


# --------------------------------------------------------------------------- #
# Killzone/session filter
# --------------------------------------------------------------------------- #

def _killzone_tag(row, cfg_sess) -> tuple[bool, str]:
    """Return (allowed, tag) for the current M1 bar's session."""
    in_lon_kz = bool(row.get('kz_london', False))
    in_ny_kz  = bool(row.get('kz_ny', False))
    in_as_kz  = bool(row.get('kz_asian', False))
    in_lon_o30 = bool(row.get('london_open_30', False))
    in_ny_o30  = bool(row.get('ny_open_30', False))
    in_off     = (row.get('session', '') == 'OFF')

    tags = []
    # Default policy:
    #   * block_off_hours=True  (champion) -> only explicit killzones are allowed,
    #     all other bars (off-hours, mid-session, between KZs) are rejected.
    #   * block_off_hours=False (RL/unrestricted --all) -> EVERY bar is admissible;
    #     killzone matches only add bonus tags for grading/BE/trailing logic.
    #     This is what lets M1 displacements at 17:00+ UTC (NY afternoon / off-
    #     hours grind-downs) refresh trigger timestamps so LIQ_SWEEP/BOS_CONT
    #     scalps fire the way Sly expects ("see everything before RL").
    allowed = not cfg_sess.block_off_hours
    if cfg_sess.trade_london_open30 and in_lon_o30:
        allowed = True; tags.append('london_open30')
    if cfg_sess.trade_ny_open30 and in_ny_o30:
        allowed = True; tags.append('ny_open30')
    if cfg_sess.trade_london_kz and in_lon_kz:
        allowed = True; tags.append('london_kz')
    if cfg_sess.trade_ny_kz and in_ny_kz:
        allowed = True; tags.append('ny_kz')
    if cfg_sess.trade_asian_range_retest and in_as_kz:
        # Asia allowed only for C-grade range-retest; caller re-checks grade
        allowed = True; tags.append('asian_kz_c_only')
    if cfg_sess.block_off_hours and in_off:
        allowed = False; tags = ['off_hours_blocked']
    return (allowed, '+'.join(tags) if tags else 'none')


# --------------------------------------------------------------------------- #
# Target selection (swing projection)
# --------------------------------------------------------------------------- #

def _runner_target(direction: int, row, cfg: ConfluenceConfig,
                   entry: float, risk: float) -> tuple[str | None, float | None]:
    """Runner target = nearest opposing major swing on a bias-aligned TF.

    We look at M15, H1, H4, D1 in that order (closest-liquidity first). The
    first one that is at least 2R away and within 8R becomes the runner target;
    if nothing qualifies we fall back to 3R (or 5R when multiple TFs align but
    are distant).
    """
    runner_min_r = 2.0
    runner_max_r = 8.0
    candidates = []
    for tf in ('M30', 'M15', 'H1', 'H4', 'D1'):
        b = row.get(f'{tf}_major_bias', 0)
        if pd.isna(b) or int(b) != direction:
            continue
        if direction == 1:
            tgt = row.get(f'{tf}_major_swing_high', np.nan)
            if pd.notna(tgt) and tgt > entry:
                dist_r = (tgt - entry) / risk if risk > 0 else np.inf
                candidates.append((tf, float(tgt), dist_r))
        else:
            tgt = row.get(f'{tf}_major_swing_low', np.nan)
            if pd.notna(tgt) and tgt < entry:
                dist_r = (entry - tgt) / risk if risk > 0 else np.inf
                candidates.append((tf, float(tgt), dist_r))
    if not candidates:
        return ('3R_fallback', None)
    # Nearest swing that is at least runner_min_r away (>= TP2 zone)
    reasonable = [c for c in candidates if runner_min_r <= c[2] <= runner_max_r]
    if reasonable:
        best = min(reasonable, key=lambda x: x[2])
        return (best[0], best[1])
    # Nothing at 2-8R: use 3R fallback
    return ('3R_fallback', None)


# --------------------------------------------------------------------------- #
# Single-bar signal evaluation
# --------------------------------------------------------------------------- #

def _evaluate_row(i: int,
                  row: pd.Series,
                  cfg: StrategyConfig,
                  # State carried across rows: dict[str, dict] tracking per-TF
                  # active (unmitigated) OB/FVG zones after a displacement.
                  state: dict,
                  *,
                  fail_trace: list[str] | None = None,
                  ) -> Signal | None:
    """Evaluate one M1 bar and return a Signal if all gates pass, else None.

    `state` is mutated across rows; it tracks the most recent unmitigated
    OBs/FVGs on each HTF, keyed by f"{tf}_{side}_{kind}" → dict(top, bot, bar).

    IMPORTANT: structural state updates (OB/FVG zone tracking, trigger
    timestamps, sweep timestamps/extremes) run on EVERY bar, BEFORE any entry
    gates are applied. Entry gates (ATR floor, session filter, zone-in-range,
    grade, risk width) only decide whether we EMIT a Signal this bar -- they
    must NOT block state tracking, or the state machine drifts blind during
    the very bars (post-rollover impulses, low-vol breakouts) where we need
    fresh triggers most. v0.9.2-v0.9.4 had this backwards: Gate 0 returned
    before state updates, causing triggers to freeze during the first 5-8
    bars after the 21:00-22:00 UTC broker rollover when ATR(14) was still
    anchored to dead-chop levels.

    Pass `fail_trace=[]` to collect human-readable reject reasons (one string
    per gate that fired, last-wins since order matters). Used by the live
    verbose loop to debug why bars that print disp/sweep flags in diagnostics
    don't produce signals; batch scan ignores it.
    """
    def _reject(msg: str):
        if fail_trace is not None:
            t = row.get('time')
            try:
                ts = pd.Timestamp(t).strftime('%H:%M')
            except Exception:
                ts = str(t)
            fail_trace.append(f"t={ts} {msg}")

    c = float(row['close'])
    atr = float(row['atr_14']) if pd.notna(row['atr_14']) else 0.0
    t_now = row['time']
    trigger_tf = cfg.trigger_tf
    retest_window = 60 if cfg.confluence.persona_gating else 120
    sweep_window = 15 if cfg.confluence.persona_gating else 30

    # ================================================================== #
    # PHASE 1: STRUCTURAL STATE UPDATE -- runs on EVERY bar, no early
    # returns here. Zones, triggers, sweeps are tracked regardless of
    # whether this bar will pass entry gates, so state stays causal.
    # ================================================================== #

    # --- 1a. Track OBs / FVGs on all OB TFs (+ trigger TF) ---
    def _update_zone(tf: str, side: str, kind: str):
        key = f"{tf}_{side}_{kind}"
        top_k = f"{tf}_{side}_{kind}_top"
        bot_k = f"{tf}_{side}_{kind}_bottom"
        mit_k = f"{tf}_{side}_{kind}_mitigated"
        top = row.get(top_k, np.nan)
        bot = row.get(bot_k, np.nan)
        mit = bool(row.get(mit_k, True))
        if pd.notna(top) and pd.notna(bot) and not mit:
            prev = state.get(key)
            if prev is None or (prev['top'] != top or prev['bot'] != bot):
                state[key] = {'top': float(top), 'bot': float(bot), 'set_at': t_now, 'fresh': True}
                state[f"{key}_entered"] = False
            else:
                prev['fresh'] = False
        elif key in state:
            state[key]['mitigated'] = True
            state[f"{key}_entered"] = True

    if atr > 0:
        # Zone tracking requires valid ATR only in the sense that features
        # need ATR to have formed zones at all; if atr<=0 zones don't exist
        # yet so skip tracking.
        track_tfs = list(dict.fromkeys(list(cfg.confluence.ob_tfs) + [trigger_tf]))
        for tf in track_tfs:
            _update_zone(tf, 'bull', 'ob')
            _update_zone(tf, 'bear', 'ob')
            _update_zone(tf, 'bull', 'fvg')
            _update_zone(tf, 'bear', 'fvg')

    # --- 1b. Displacement/BOS trigger freshness ---
    # Trigger freshness: any bullish structural impulse (disp/BOS/CHoCH on
    # M1 or trigger TF) resets the trigger window. v0.9.5 missed the minor
    # CHoCH variants at M1 level and the minor/major CHoCH variants at the
    # trigger-TF level (M5), causing the engine to ignore legitimate CHoCH-
    # driven reversals like the 22:27+ impulse Sly watched.
    bull_trig_fresh = (bool(row.get(f'{trigger_tf}_bull_disp', False)) or
                       bool(row.get(f'{trigger_tf}_minor_bos_up', False)) or
                       bool(row.get(f'{trigger_tf}_major_bos_up', False)) or
                       bool(row.get(f'{trigger_tf}_minor_choch_up', False)) or
                       bool(row.get(f'{trigger_tf}_major_choch_up', False)) or
                       bool(row.get('bull_disp', False)) or
                       bool(row.get('minor_bos_up', False)) or
                       bool(row.get('major_bos_up', False)) or
                       bool(row.get('minor_choch_up', False)) or
                       bool(row.get('major_choch_up', False)))
    bear_trig_fresh = (bool(row.get(f'{trigger_tf}_bear_disp', False)) or
                       bool(row.get(f'{trigger_tf}_minor_bos_dn', False)) or
                       bool(row.get(f'{trigger_tf}_major_bos_dn', False)) or
                       bool(row.get(f'{trigger_tf}_minor_choch_dn', False)) or
                       bool(row.get(f'{trigger_tf}_major_choch_dn', False)) or
                       bool(row.get('bear_disp', False)) or
                       bool(row.get('minor_bos_dn', False)) or
                       bool(row.get('major_bos_dn', False)) or
                       bool(row.get('minor_choch_dn', False)) or
                       bool(row.get('major_choch_dn', False)))
    # Detect opposite-CHoCH transitions BEFORE updating triggers so we can
    # use them to reset the BOS_CONT one-shot flag below.
    bull_choch_now = (bool(row.get(f'{trigger_tf}_minor_choch_up', False)) or
                      bool(row.get(f'{trigger_tf}_major_choch_up', False)) or
                      bool(row.get('minor_choch_up', False)) or
                      bool(row.get('major_choch_up', False)))
    bear_choch_now = (bool(row.get(f'{trigger_tf}_minor_choch_dn', False)) or
                      bool(row.get(f'{trigger_tf}_major_choch_dn', False)) or
                      bool(row.get('minor_choch_dn', False)) or
                      bool(row.get('major_choch_dn', False)))

    bull_trigger_this_bar = False
    bear_trigger_this_bar = False
    if bull_trig_fresh:
        state['_last_bull_trigger'] = t_now
        bull_trigger_this_bar = True
        # m5_bull = any bullish structural impulse on the trigger TF (M5).
        # v0.9.5 omitted minor_choch_up / major_choch_up here, which meant
        # a bullish M5 CHoCH (which IS a valid fresh M5 trigger for retest
        # zones) refreshed the M1-level window but NOT the _m5 window —
        # causing RETEST longs to miss the M5 retest window by ~60 bars.
        m5_bull = bool(row.get(f'{trigger_tf}_bull_disp', False)) or \
                 bool(row.get(f'{trigger_tf}_minor_bos_up', False)) or \
                 bool(row.get(f'{trigger_tf}_major_bos_up', False)) or \
                 bool(row.get(f'{trigger_tf}_minor_choch_up', False)) or \
                 bool(row.get(f'{trigger_tf}_major_choch_up', False))
        if m5_bull:
            state['_last_bull_trigger_m5'] = t_now
    if bear_trig_fresh:
        state['_last_bear_trigger'] = t_now
        bear_trigger_this_bar = True
        m5_bear = bool(row.get(f'{trigger_tf}_bear_disp', False)) or \
                 bool(row.get(f'{trigger_tf}_minor_bos_dn', False)) or \
                 bool(row.get(f'{trigger_tf}_major_bos_dn', False)) or \
                 bool(row.get(f'{trigger_tf}_minor_choch_dn', False)) or \
                 bool(row.get(f'{trigger_tf}_major_choch_dn', False))
        if m5_bear:
            state['_last_bear_trigger_m5'] = t_now

    # BOS_CONT one-shot: reset "entered" flag when an opposite CHoCH fires
    # (structure flip = new leg, so the next same-direction BOS break is a
    # fresh setup). Without this, one bear BOS entry blocks the next bear
    # BOS for the rest of the trend even after a CHoCH-up pullback and a
    # fresh BOS-down continuation.
    if bull_choch_now:
        state['_bos_entered_-1'] = False   # bear BOS arm re-armed by bull CHoCH
    if bear_choch_now:
        state['_bos_entered_+1'] = False   # bull BOS arm re-armed by bear CHoCH

    # --- 1c. Liquidity sweep tracking ---
    # Track sweep events AND invalidate them if price penetrates beyond the
    # wick extreme (i.e. the "stop-run" didn't reverse — it turned into a
    # continuation). v0.9.6 only checked current bar close vs sweep_px, which
    # let a bull sweep at 4597 get smashed through to 4594 then fire a bad long
    # when price rallied back to 4600 — the reversal thesis was already dead
    # the moment price closed 3+ points below the sweep wick.
    low = float(row.get('low', c))
    high = float(row.get('high', c))
    for direction in (1, -1):
        side = 'bull' if direction == 1 else 'bear'
        m1_sweep_now = bool(row.get('bull_liq_sweep' if direction==1 else 'bear_liq_sweep', False))
        tf_sweep_now = bool(row.get(f'{trigger_tf}_{"bull" if direction==1 else "bear"}_liq_sweep', False))
        sweep_key_ts = f'_last_{side}_sweep_ts'
        sweep_key_px = f'_last_{side}_sweep_px'
        sweep_key_inv = f'_last_{side}_sweep_invaded'
        if m1_sweep_now or tf_sweep_now:
            sweep_px = np.nan
            if m1_sweep_now:
                sweep_px = row.get('bull_sweep_px' if direction==1 else 'bear_sweep_px', np.nan)
            if pd.isna(sweep_px) and tf_sweep_now:
                sweep_px = row.get(f'{trigger_tf}_{"bull" if direction==1 else "bear"}_sweep_px', np.nan)
            if pd.notna(sweep_px):
                state[sweep_key_ts] = t_now
                state[sweep_key_px] = float(sweep_px)
                # New sweep resets invalidation
                state[sweep_key_inv] = False
        # Check if price has violated the sweep extreme since the sweep was set.
        # A bull sweep is a wick below a low — if price subsequently makes a
        # lower LOW more than 0.5 ATR below the sweep wick, the stop-run failed
        # (turning into continuation) and the reversal setup is dead until a
        # fresh sweep fires. Bear sweep symmetric.
        sweep_px_cur = state.get(sweep_key_px)
        if sweep_px_cur is not None and not state.get(sweep_key_inv, False):
            inv_buffer = 0.5 * atr if atr > 0 else 0.5
            if direction == 1 and low < float(sweep_px_cur) - inv_buffer:
                state[sweep_key_inv] = True
            elif direction == -1 and high > float(sweep_px_cur) + inv_buffer:
                state[sweep_key_inv] = True

    # ================================================================== #
    # PHASE 2: ENTRY GATES -- applied to decide whether to emit a Signal
    # this bar. If we bail here, state has already been updated above.
    # ================================================================== #

    if atr <= 0:
        _reject(f"atr<=0 atr={atr}"); return None

    # Gate 0: ATR sanity. Champion uses tight band [0.0004, 0.02] = ~$1.0
    # min ATR at $2500 XAU. In unrestricted/--all mode we lower the floor
    # to ~$0.50 at $4600 so the first 5-8 bars after the broker rollover
    # (where Wilder ATR14 is still anchored to dead-chop from before the
    # break) are not rejected -- those are exactly the impulse bars that
    # start the new move. The upper cap at $90 ATR (0.02) still filters
    # FOMC/news flash-crashes.
    atr_pct = atr / c if c > 0 else 0.0
    min_atr_pct_gate = cfg.confluence.min_atr_pct if cfg.confluence.persona_gating else 0.00010
    if atr_pct < min_atr_pct_gate or atr_pct > cfg.confluence.max_atr_pct:
        _reject(f"atr_pct={atr_pct:.5f} outside band [{min_atr_pct_gate},{cfg.confluence.max_atr_pct}] atr={atr:.2f} c={c:.2f}")
        return None

    # Gate 1: session / killzone
    allowed, kz_tag = _killzone_tag(row, cfg.sessions)
    if not allowed:
        _reject(f"session blocked kz={kz_tag}"); return None

    # Window checks (use post-update state so a trigger firing THIS bar
    # opens the window immediately rather than waiting one bar).
    def _within_window(ts):
        if ts is None: return False
        return (t_now - ts) <= pd.Timedelta(minutes=retest_window)

    bull_window = _within_window(state.get('_last_bull_trigger'))
    bear_window = _within_window(state.get('_last_bear_trigger'))
    bull_window_m5 = _within_window(state.get('_last_bull_trigger_m5'))
    bear_window_m5 = _within_window(state.get('_last_bear_trigger_m5'))

    candidates: list[int] = []
    # Directional filter — when persona_gating is OFF (RL training mode), both
    # sides are emitted regardless of accept_longs/accept_shorts so the agent
    # sees the full action space. When gating is ON (default paper-trade
    # persona), we honor the static directional filter.
    if bull_window and (cfg.confluence.accept_longs or not cfg.confluence.persona_gating):
        candidates.append(1)
    if bear_window and (cfg.confluence.accept_shorts or not cfg.confluence.persona_gating):
        candidates.append(-1)

    # ================================================================== #
    # SETUP A: LIQUIDITY SWEEP SCALP (stop-run reversal)
    # Wick takes out a recent minor swing (sellside/buyside liquidity),
    # closes back inside, then displacement fires in the reversal
    # direction within `sweep_window` M1 bars. Enter at displacement
    # close, SL just beyond the sweep wick extreme, quick 0.85R TP.
    # ================================================================== #
    # sweep_window is already set above (Phase 1); reuse it.
    for direction in (1, -1):
        if direction not in candidates:
            continue
        side = 'bull' if direction == 1 else 'bear'
        last_sweep_ts = state.get(f'_last_{side}_sweep_ts')
        if last_sweep_ts is None:
            if fail_trace is not None:
                _reject(f"LIQ_SWEEP {side}: no sweep in state")
            continue
        age_min = (t_now - last_sweep_ts).total_seconds() / 60.0
        if age_min > sweep_window:
            if fail_trace is not None:
                _reject(f"LIQ_SWEEP {side}: sweep age {age_min:.0f}m > window {sweep_window}m")
            continue
        sweep_extreme = state.get(f'_last_{side}_sweep_px')
        if sweep_extreme is None:
            continue
        # Per-bar invalidation flag: if price penetrated more than 0.5 ATR
        # through the sweep wick after the sweep fired, the reversal is dead
        # (the stop-run turned into continuation) and we do NOT enter even
        # if close is back above the wick on a later rally. This is the bug
        # that cost ~53 ZAR on the 07:29 long (price crashed to 4594 -- 3
        # points below the 4597 sweep -- then rallied to 4600 and fired long
        # into the bear continuation).
        if state.get(f'_last_{side}_sweep_invaded', False):
            if fail_trace is not None:
                _reject(f"LIQ_SWEEP {side}: sweep penetrated post-sweep (failed reversal, now continuation)")
            continue
        # Current bar close must still be on the reversal side of the wick.
        if direction == 1 and c < float(sweep_extreme):
            if fail_trace is not None:
                _reject(f"LIQ_SWEEP {side}: c={c:.2f} < sweep_px={float(sweep_extreme):.2f} (failed reversal)")
            continue
        if direction == -1 and c > float(sweep_extreme):
            if fail_trace is not None:
                _reject(f"LIQ_SWEEP {side}: c={c:.2f} > sweep_px={float(sweep_extreme):.2f} (failed reversal)")
            continue

        # v0.9.8 PROXIMITY GATE: close must be within 2.0 ATR of the sweep wick.
        # A liquidity-sweep scalp fires on the REJECTION immediately after the
        # wick — not after price has already run 10-18 points in the reversal
        # direction. The 09:36 bear sweep at wick=4599 entered SHORT at fill
        # =4580.97 (18 points below the wick, ~7.2 ATR away after ATR expanded
        # on the crash bar) — pure chase into a waterfall, not a scalp. The
        # existing risk<7ATR gate failed because ATR itself expanded to ~2.6 on
        # the crash bar, letting 18 points slip through as "6.9 ATR". Require
        # close within 2 ATR of the wick so we only take the first bar or two
        # of rejection, not the whole continuation.
        dist_from_wick = abs(c - float(sweep_extreme))
        if atr > 0 and dist_from_wick > 2.0 * atr:
            if fail_trace is not None:
                _reject(f"LIQ_SWEEP {side}: c={c:.2f} too far from sweep_px={float(sweep_extreme):.2f} "
                        f"(dist={dist_from_wick:.2f} > 2ATR={2.0*atr:.2f}) — chasing")
            continue
        # Require displacement/BOS/CHoCH in reversal direction (M1 or trigger_tf).
        # v0.9.5 only checked disp and major/minor BOS; CHoCH breaks (which
        # are the primary signal after a liquidity grab) were silently
        # skipped -- added them here so a fresh sweep+CHoCH reversal fires.
        m1_disp = bool(row.get('bull_disp' if direction==1 else 'bear_disp', False))
        tf_disp = bool(row.get(f'{trigger_tf}_{"bull" if direction==1 else "bear"}_disp', False))
        m1_bos = bool(row.get('minor_bos_up' if direction==1 else 'minor_bos_dn', False)) or \
                 bool(row.get('major_bos_up' if direction==1 else 'major_bos_dn', False)) or \
                 bool(row.get('minor_choch_up' if direction==1 else 'minor_choch_dn', False)) or \
                 bool(row.get('major_choch_up' if direction==1 else 'major_choch_dn', False))
        vol_surge = bool(row.get('vol_spike', False)) or bool(row.get(f'{trigger_tf}_vol_spike', False))
        if cfg.confluence.persona_gating:
            if not (m1_disp or tf_disp or (m1_bos and vol_surge)):
                if fail_trace is not None:
                    _reject(f"LIQ_SWEEP {side}: no disp/BOS confirmation (champion requires vol_surge)")
                continue
        else:
            if not (m1_disp or tf_disp or m1_bos):
                if fail_trace is not None:
                    _reject(f"LIQ_SWEEP {side}: no disp/BOS/CHoCH confirmation yet")
                continue
        # One entry per sweep event (dedupe by timestamp)
        sweep_key = f"_swept_{direction}_{last_sweep_ts.isoformat()}"
        if state.get(sweep_key):
            continue
        state[sweep_key] = True

        entry = c
        # LIQ_SWEEP SL buffer: 0.30 ATR past the wick (not 0.075). A liquidity
        # sweep is a stop-run rejection — price frequently retests the wick by
        # 0.10-0.25 ATR (20-50c on XAU) before reversing. v0.9.9- used
        # ob_invalidation_buffer*1.5 = 0.075 ATR (~14c) which got hunted
        # instantly on the 11:55 bear sweep (fill 4598.79, wick 4600.07, SL
        # 4600.21 — price wick-retested to 4599.8 and clipped).
        sweep_sl_buffer = max(0.30 * atr, 0.50)
        if direction == 1:
            stop = float(sweep_extreme) - sweep_sl_buffer
        else:
            stop = float(sweep_extreme) + sweep_sl_buffer
        risk = abs(entry - stop)
        if risk <= 0 or risk < 0.5 * atr or risk > 7.0 * atr or risk > 40.0:
            continue
        risk_atr = risk / atr if atr > 0 else 0
        if cfg.confluence.persona_gating and risk_atr < cfg.confluence.min_risk_atr:
            continue

        grade, conf_tags = _grade(direction, row, cfg.confluence,
                                  bonus_killzone=('+' in kz_tag or 'open30' in kz_tag))
        if grade == 'fail':
            if cfg.confluence.persona_gating:
                continue
            grade = 'C'
            conf_tags = ['scalp_fallback_c']
        elif cfg.confluence.persona_gating and grade not in cfg.confluence.accept_grades:
            continue
        # Scalp grades cap: LIQ_SWEEP is a quick reversal scalp against a
        # fresh wick -- HTF bias can be laggy right after a sweep (the D1/H4/
        # H1 bias was still bull on the 07:29 bar even though price had been
        # selling off for 25 minutes straight, leading to an A+ grade on a
        # losing scalp). Cap LIQ_SWEEP at B in champion mode, at C in
        # unrestricted mode. Retain A+/A for retest setups only.
        if not cfg.confluence.persona_gating:
            if grade in ('A+', 'A', 'B'):
                conf_tags.append(f'downgraded_from_{grade}')
                grade = 'C'
        elif grade == 'A+':
            conf_tags.append('downgraded_from_A+')
            grade = 'B'

        if direction == 1:
            tp1 = entry + cfg.exits.tp1_r * risk
        else:
            tp1 = entry - cfg.exits.tp1_r * risk
        tp2 = tp1
        tgt_tf, tgt_px = _runner_target(direction, row, cfg.confluence, entry, risk)
        if tgt_px is None:
            tgt_px = entry + direction * 2.0 * risk
            tgt_tf = '2R_scalp'
        risk_pct = cfg.grades.risk_fraction(grade) * 0.5

        tags = [f'trigger_{trigger_tf}', 'setup_LIQ_SWEEP'] + conf_tags
        tags.append(f'grade_{grade}')
        tags.append(f'kz_{kz_tag}')
        if vol_surge:
            tags.append('vol_surge')
        bias_summary = {}
        for tf in ('W1','D1','H4','H1','M30','M15','M5'):
            b = row.get(f'{tf}_major_bias', 0)
            if pd.notna(b): bias_summary[tf] = int(b)

        sig = Signal(
            time=t_now, direction=direction, entry=entry, stop=stop,
            tp1=tp1, tp2=tp2, tp_runner=float(tgt_px), risk_per_unit=risk,
            grade=grade, risk_pct=risk_pct, setup_kind="LIQ_SWEEP",
            confluence=tags, swing_target_tf=tgt_tf, swing_target_price=float(tgt_px),
            atr_at_entry=atr, htf_bias_summary=bias_summary,
            session=str(row.get('session','')), killzone=kz_tag,
        )
        return sig

    # ================================================================== #
    # SETUP B: BOS / CHoCH CONTINUATION SCALP
    # Minor/major BOS or CHoCH fires with displacement — ride the impulse,
    # don't wait for retest. SL below last opposing swing, TP at 0.85R.
    # v0.9.6 adds M1 CHoCH and trigger-TF minor CHoCH detection (v0.9.5
    # silently ignored minor_choch on either TF — the bar Sly watched at
    # 22:27 with `minor_choch_up` printed but sigs=0 was this bug).
    # ================================================================== #
    for direction in (1, -1):
        if direction not in candidates:
            continue
        side = 'bull' if direction == 1 else 'bear'
        # Fresh BOS/CHoCH on trigger_tf OR M1 in this direction.
        # BOS = continuation of existing structure; CHoCH = reversal.
        # Both are valid continuation-scalp entries right after the break.
        if direction == 1:
            tf_structure = (bool(row.get(f'{trigger_tf}_minor_bos_up', False)) or
                            bool(row.get(f'{trigger_tf}_major_bos_up', False)) or
                            bool(row.get(f'{trigger_tf}_minor_choch_up', False)) or
                            bool(row.get(f'{trigger_tf}_major_choch_up', False)))
            m1_structure = (bool(row.get('minor_bos_up', False)) or
                            bool(row.get('major_bos_up', False)) or
                            bool(row.get('minor_choch_up', False)) or
                            bool(row.get('major_choch_up', False)))
        else:
            tf_structure = (bool(row.get(f'{trigger_tf}_minor_bos_dn', False)) or
                            bool(row.get(f'{trigger_tf}_major_bos_dn', False)) or
                            bool(row.get(f'{trigger_tf}_minor_choch_dn', False)) or
                            bool(row.get(f'{trigger_tf}_major_choch_dn', False)))
            m1_structure = (bool(row.get('minor_bos_dn', False)) or
                            bool(row.get('major_bos_dn', False)) or
                            bool(row.get('minor_choch_dn', False)) or
                            bool(row.get('major_choch_dn', False)))
        if not (tf_structure or m1_structure):
            if fail_trace is not None:
                _reject(f"BOS_CONT {side}: no BOS/CHoCH structure on this bar")
            continue
        # Momentum confirmation. Champion (persona_gating=True) requires
        # displacement AND volume spike (PF 2.00 preservation); unrestricted
        # --all mode relaxes this to just the structural break above
        # (disp/vol are bonus tags, not gates) so we don't miss low-volume
        # grind continuations.
        disp_flag = f'{trigger_tf}_bull_disp' if direction == 1 else f'{trigger_tf}_bear_disp'
        m1_disp_flag = 'bull_disp' if direction == 1 else 'bear_disp'
        has_disp = bool(row.get(disp_flag, False)) or bool(row.get(m1_disp_flag, False))
        has_vol = bool(row.get(f'{trigger_tf}_vol_spike', False)) or bool(row.get('vol_spike', False))
        if cfg.confluence.persona_gating:
            if not (has_disp and has_vol):
                if fail_trace is not None:
                    _reject(f"BOS_CONT {side}: persona requires disp+vol_surge")
                continue

        # ONE-SHOT PER STRUCTURAL LEG (v0.9.8 fix): the ATR-ZigZag edge-flag
        # (sh_broken/sl_broken) resets every time a new swing pivot forms, so
        # during a strong impulse (e.g. the 08:34 London bear leg that dropped
        # from 4603→4595 in 5 minutes) a fresh minor_bos_dn fires on MULTIPLE
        # consecutive bars as each successive lower swing-low gets broken.
        # Without this guard we entered 5 SHORT BOS_CONT positions (tickets
        # 3145423145..3145441280) on the same bear leg — piling in to what
        # should have been ONE scalp. We now allow AT MOST one BOS_CONT entry
        # per direction per leg, reset only by an opposite CHoCH (detected
        # above in Phase 1b).
        #
        # CALLER RESPONSIBILITY: the one-shot arm key (`_bos_entered_{dir}`)
        # is SET BY THE CALLER (live trader / backtest engine) AFTER a fill
        # is confirmed. We only CHECK it here — _evaluate_row never sets it,
        # so state-priming warmups walking historical bars don't prematurely
        # arm the key. Without this, a bot restart mid-leg would see the
        # historical BOS bar during _prime_state, set the key as a side
        # effect, and block the next legitimate entry after restart.
        bos_arm_key = f'_bos_entered_{direction:+d}'
        if state.get(bos_arm_key, False):
            if fail_trace is not None:
                _reject(f"BOS_CONT {side}: already entered this leg (one-shot until opposite CHoCH)")
            continue

        # BOS_CONT fires ONLY on a bar where the trigger timestamp was JUST
        # refreshed (i.e. a fresh structural impulse occurred THIS bar) AND
        # a BOS/CHoCH structural break is present THIS BAR. The previous
        # behaviour allowed re-entry on bars where bear_window was open and
        # minor_bos_dn was "still hot" from a prior bar — which combined
        # with the ATR-ZigZag edge-reset bug to pyramid entries.
        trigger_fresh_this_bar = (bull_trigger_this_bar if direction == 1 else bear_trigger_this_bar)
        structure_now = tf_structure or m1_structure
        if not (trigger_fresh_this_bar and structure_now):
            if fail_trace is not None:
                _reject(f"BOS_CONT {side}: no fresh BOS/CHoCH this bar (stale trigger)")
            continue

        # SL anchor: last opposing minor swing (the level just broken FROM)
        if direction == 1:
            sl_anchor = row.get(f'{trigger_tf}_minor_swing_low', np.nan)
            if pd.isna(sl_anchor): sl_anchor = row.get('minor_swing_low', np.nan)
        else:
            sl_anchor = row.get(f'{trigger_tf}_minor_swing_high', np.nan)
            if pd.isna(sl_anchor): sl_anchor = row.get('minor_swing_high', np.nan)
        if pd.isna(sl_anchor):
            if fail_trace is not None:
                _reject(f"BOS_CONT {side}: no minor swing anchor for SL")
            continue
        entry = c
        buffer_amt = cfg.exits.ob_invalidation_buffer * atr
        if direction == 1:
            stop = float(sl_anchor) - buffer_amt
        else:
            stop = float(sl_anchor) + buffer_amt
        risk = abs(entry - stop)
        if risk <= 0 or risk < 0.5 * atr or risk > 7.0 * atr or risk > 40.0:
            if fail_trace is not None:
                _reject(f"BOS_CONT {side}: risk={risk:.2f} atr={atr:.2f} outside bounds (0.5-7ATR, <=$40)")
            continue
        risk_atr = risk / atr if atr > 0 else 0

        grade, conf_tags = _grade(direction, row, cfg.confluence,
                                  bonus_killzone=('+' in kz_tag or 'open30' in kz_tag))
        if grade == 'fail':
            if cfg.confluence.persona_gating:
                continue
            grade = 'C'
            conf_tags = ['scalp_fallback_c']
        elif cfg.confluence.persona_gating and grade not in cfg.confluence.accept_grades:
            # Champion persona only: only BOS_CONT with A+/A/B grade
            continue
        # Scalp grade cap for BOS_CONT (same reasoning as LIQ_SWEEP: M1 impulse
        # scalps against laggy HTF bias must never size as A+/A in unrestricted
        # mode, and cap at B in champion mode).
        if not cfg.confluence.persona_gating:
            if grade in ('A+', 'A', 'B'):
                conf_tags.append(f'downgraded_from_{grade}')
                grade = 'C'
        elif grade == 'A+':
            conf_tags.append('downgraded_from_A+')
            grade = 'B'

        if direction == 1:
            tp1 = entry + cfg.exits.tp1_r * risk
        else:
            tp1 = entry - cfg.exits.tp1_r * risk
        tp2 = tp1
        tgt_tf, tgt_px = _runner_target(direction, row, cfg.confluence, entry, risk)
        if tgt_px is None:
            tgt_px = entry + direction * 2.0 * risk
            tgt_tf = '2R_scalp'
        # BOS continuation scalps are 60% size (quick in-out momentum ride).
        risk_pct = cfg.grades.risk_fraction(grade) * 0.6

        tags = [f'trigger_{trigger_tf}', 'setup_BOS_CONT'] + conf_tags
        tags.append(f'grade_{grade}')
        tags.append(f'kz_{kz_tag}')
        if has_vol:
            tags.append('vol_spike')
        bias_summary = {}
        for tf in ('W1','D1','H4','H1','M30','M15','M5'):
            b = row.get(f'{tf}_major_bias', 0)
            if pd.notna(b): bias_summary[tf] = int(b)

        sig = Signal(
            time=t_now, direction=direction, entry=entry, stop=stop,
            tp1=tp1, tp2=tp2, tp_runner=float(tgt_px), risk_per_unit=risk,
            grade=grade, risk_pct=risk_pct, setup_kind="BOS_CONT",
            confluence=tags, swing_target_tf=tgt_tf, swing_target_price=float(tgt_px),
            atr_at_entry=atr, htf_bias_summary=bias_summary,
            session=str(row.get('session','')), killzone=kz_tag,
        )
        # NOTE: caller (live trader / backtest) is responsible for setting
        # state[bos_arm_key] = True AFTER a successful fill — not here.
        return sig

    if not candidates:
        return None

    for direction in candidates:
        side = 'bull' if direction == 1 else 'bear'

        # Zone retest trades (RETEST_OB / RETEST_FVG) REQUIRE an M5 trigger to
        # have fired in the window — without an M5 displacement, no M5 OB/FVG
        # zone has formed, so any "entry" would be a phantom. This preserves
        # the v0.9.0 champion PF 2.00 baseline while letting M1 displacements
        # drive the LIQ_SWEEP / BOS_CONT scalp setups above.
        m5_window_ok = bull_window_m5 if direction == 1 else bear_window_m5
        if not m5_window_ok:
            continue

        # Gate 3: find an active unmitigated zone (OB or FVG) on an OB TF whose
        # price region contains (or was just tapped by) current close.
        zone = None
        for tf in cfg.confluence.ob_tfs:
            # Prefer OB over FVG, prefer higher TF in ob_tfs order
            for kind in ('ob', 'fvg'):
                key = f"{tf}_{side}_{kind}"
                z = state.get(key)
                if z is None or z.get('mitigated'):
                    continue
                if state.get(f"{key}_entered"):
                    continue  # already took an entry on this zone
                top, bot = z['top'], z['bot']
                # Price must be "in" the zone (close within [bot, top])
                if bot <= c <= top:
                    zone = (kind.upper(), tf, top, bot)
                    break
            if zone: break
        if zone is None:
            continue

        kind, ztf, ztop, zbot = zone

        # Gate 4: zone kind / TF gating (battle-tested filter).
        # When persona_gating is OFF (RL mode), we accept all discovered zones
        # on any configured OB TF — the agent learns which to take/skip.
        if cfg.confluence.persona_gating:
            if kind.upper() not in cfg.confluence.accept_zone_kinds:
                continue
            if ztf not in cfg.confluence.accept_ob_tfs:
                continue

        # Gate 5: HTF bias agreement + grade assignment.
        # _grade() always returns a grade ("A+"/"A"/"B"/"C"/"fail"). In persona
        # mode we reject grades outside accept_grades; in RL mode we emit
        # everything except hard structural fails (grade=="fail" means HTF
        # CHoCH against with zero supporting confluence — not a real setup).
        grade, conf_tags = _grade(direction, row, cfg.confluence,
                                  bonus_killzone=('+' in kz_tag or 'open30' in kz_tag))
        if grade == 'fail':
            continue
        if cfg.confluence.persona_gating and grade not in cfg.confluence.accept_grades:
            continue

        # Asia entries: C-grade only (range-retest, never A+/A/B). Downgrade if enabled.
        if 'asian_kz_c_only' in kz_tag and cfg.sessions.trade_asian_range_retest and grade != 'C':
            grade = 'C'
            conf_tags.append('asia_downgraded_to_c')

        # Gate 5: emergency CHoCH on emergency_choch_tf AGAINST direction blocks entry
        emerg_tf = cfg.exits.emergency_choch_tf
        if direction == 1 and bool(row.get(f'{emerg_tf}_major_choch_dn', False)):
            continue
        if direction == -1 and bool(row.get(f'{emerg_tf}_major_choch_up', False)):
            continue

        # Gate 6: avoid chasing — don't enter if price is at/below OB TOP on a
        # bull entry (i.e. already blew through) — we are looking for the
        # retest, not the break. We have already ensured c is within the zone.
        # Additionally skip if the zone was just set on THIS same M1 bar (price
        # hasn't retraced yet).
        zset = state.get(f"{ztf}_{side}_{kind}", {}).get('set_at')
        if zset is not None and zset == t_now:
            continue  # wait for retest

        # ---- Compute SL/TP/entry ----
        entry = c
        buffer_amt = cfg.exits.ob_invalidation_buffer * atr
        if direction == 1:
            stop = zbot - buffer_amt
        else:
            stop = ztop + buffer_amt
        risk = abs(entry - stop)
        if risk <= 0 or risk < 0.3 * atr:  # require reasonable risk distance
            continue
        if risk > 7.0 * atr:
            continue  # too-wide stop = zone is far away / news spike; likely stale
        # Absolute cap: for XAUUSD, $50+ risk per oz is almost always a stale/
        # broken zone that already mitigated on another TF. (Layer 5 sizing
        # will also clamp per-account risk, but we filter bad zones here.)
        if risk > 50.0:
            continue

        # Gate 6: risk-in-ATR width filter. In persona mode reject setups outside
        # the configured band; in RL mode tag them but keep all setups within
        # sane structural bounds (0.3–7 ATR) so the agent learns sizing/SL.
        risk_atr = risk / atr if atr > 0 else 0.0
        if cfg.confluence.persona_gating:
            if risk_atr < cfg.confluence.min_risk_atr:
                continue
            if risk_atr > cfg.confluence.max_risk_atr:
                continue
        else:
            if risk_atr < 0.3 or risk_atr > 7.0:
                continue

        if direction == 1:
            tp1 = entry + cfg.exits.tp1_r * risk
            tp2 = entry + cfg.exits.tp2_r * risk if cfg.exits.tp2_pct > 0 else entry + cfg.exits.tp1_r * risk
        else:
            tp1 = entry - cfg.exits.tp1_r * risk
            tp2 = entry - cfg.exits.tp2_r * risk if cfg.exits.tp2_pct > 0 else entry - cfg.exits.tp1_r * risk

        tgt_tf, tgt_px = _runner_target(direction, row, cfg.confluence, entry, risk)
        if tgt_px is None:
            tgt_px = entry + direction * 3.0 * risk
            tgt_tf = '3R_fallback'

        risk_pct = cfg.grades.risk_fraction(grade)

        tags = [f'trigger_{trigger_tf}', f'zone_{kind}_on_{ztf}'] + conf_tags
        tags.append(f'grade_{grade}')
        tags.append(f'kz_{kz_tag}')
        if bool(row.get('vol_spike', False)):
            tags.append('vol_spike')

        bias_summary = {}
        for tf in ('W1','D1','H4','H1','M30','M15','M5'):
            b = row.get(f'{tf}_major_bias', 0)
            if pd.notna(b):
                bias_summary[tf] = int(b)

        # `kind` here is uppercase ("OB"/"FVG"); normalize for state key.
        kind_key = kind.lower()

        setup_kind_label = "RETEST_OB" if kind == "OB" else "RETEST_FVG"
        sig = Signal(
            time=t_now,
            direction=direction,
            entry=entry,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            tp_runner=float(tgt_px),
            risk_per_unit=risk,
            grade=grade,
            risk_pct=risk_pct,
            setup_kind=setup_kind_label,
            confluence=tags,
            ob_tf=ztf if kind == 'OB' else None,
            ob_top=ztop if kind == 'OB' else None,
            ob_bottom=zbot if kind == 'OB' else None,
            fvg_top=ztop if kind == 'FVG' else None,
            fvg_bottom=zbot if kind == 'FVG' else None,
            swing_target_tf=tgt_tf,
            swing_target_price=float(tgt_px),
            atr_at_entry=atr,
            htf_bias_summary=bias_summary,
            session=str(row.get('session', '')),
            killzone=kz_tag,
        )
        # Mark the zone as "entered" so we don't fire again on the same zone
        # retest until a new zone replaces it.
        state[f"{ztf}_{side}_{kind_key}_entered"] = True
        return sig

    return None


# --------------------------------------------------------------------------- #
# Batch scan
# --------------------------------------------------------------------------- #

def _strategy_columns(cfg: StrategyConfig) -> list[str]:
    """Return the list of columns actually needed by the strategy engine."""
    cols = ['time', 'open', 'high', 'low', 'close', 'atr_14', 'session',
            'kz_asian', 'kz_london', 'kz_ny', 'london_open_30', 'ny_open_30',
            'vol_spike',
            'bull_disp', 'bear_disp',
            'minor_bos_up', 'minor_bos_dn', 'minor_choch_up', 'minor_choch_dn',
            'major_bos_up', 'major_bos_dn', 'major_choch_up', 'major_choch_dn',
            'bull_liq_sweep', 'bear_liq_sweep', 'bull_sweep_px', 'bear_sweep_px',
            'minor_swing_high', 'minor_swing_low']
    tfs = set(cfg.confluence.ob_tfs) | set(cfg.confluence.a_plus_required_tfs) | \
          set(cfg.confluence.a_required_tfs) | set(cfg.confluence.b_required_tfs) | \
          set(cfg.confluence.c_required_tfs)
    tfs.add(cfg.confluence.pd_range_tf)
    tfs.add(cfg.trigger_tf)
    tfs.add(cfg.exits.emergency_choch_tf)
    tfs.add(cfg.exits.runner_stop_tf)
    tfs.add('D1'); tfs.add('H4'); tfs.add('H1')  # always needed for runner targets
    for tf in tfs:
        prefix = '' if tf == 'M1' else f'{tf}_'
        cols += [
            f'{prefix}major_bias', f'{prefix}major_swing_high', f'{prefix}major_swing_low',
            f'{prefix}price_in_range_pct',
            f'{prefix}bull_disp', f'{prefix}bear_disp',
            f'{prefix}minor_bos_up', f'{prefix}minor_bos_dn',
            f'{prefix}minor_choch_up', f'{prefix}minor_choch_dn',
            f'{prefix}major_bos_up', f'{prefix}major_bos_dn',
            f'{prefix}major_choch_up', f'{prefix}major_choch_dn',
            f'{prefix}bull_liq_sweep', f'{prefix}bear_liq_sweep',
            f'{prefix}bull_sweep_px', f'{prefix}bear_sweep_px',
            f'{prefix}bull_ob_top', f'{prefix}bull_ob_bottom', f'{prefix}bull_ob_mitigated',
            f'{prefix}bear_ob_top', f'{prefix}bear_ob_bottom', f'{prefix}bear_ob_mitigated',
            f'{prefix}bull_fvg_top', f'{prefix}bull_fvg_bottom', f'{prefix}bull_fvg_mitigated',
            f'{prefix}bear_fvg_top', f'{prefix}bear_fvg_bottom', f'{prefix}bear_fvg_mitigated',
        ]
    # Dedupe preserving order
    seen = set(); out = []
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def _load_aligned_partitions(root: Path, symbol: str,
                             columns: list[str] | None = None) -> pd.DataFrame:
    """Load aligned M1 partitions into one frame (month-by-month concat)."""
    from ..data.storage import discover_partitions
    base = root / f"symbol={symbol}" / "timeframe=M1"
    files = discover_partitions(base, "**/*.parquet")
    if not files:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for f in sorted(files):
        try:
            frames.append(pd.read_parquet(f, columns=columns))
        except Exception:
            # If columns don't all exist (e.g. schema mismatch), fall back to all
            frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in df.columns:
        if col.endswith("_bar_time"):
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)
    return df


def scan(df: pd.DataFrame,
         cfg: StrategyConfig | None = None,
         progress: Callable[[str], None] | None = None) -> list[Signal]:
    """Scan an M1-aligned DataFrame and return a list of Signal objects.

    `df` must be sorted by `time` ascending and contain the full set of
    M1 + HTF-prefixed columns produced by Layer 3.

    State (active OB/FVG zones, last-trigger timestamps, entry flags) is
    carried across rows via a plain dict, mirroring the streaming scanner
    in `scanner.scan_aligned()`.
    """
    cfg = cfg or StrategyConfig()
    progress = progress or (lambda _m: None)

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time', kind='mergesort').reset_index(drop=True)

    signals: list[Signal] = []
    n = len(df)
    # 3000 M1 bars (~2 trading days) for indicator warmup — long enough for EMA200,
    # ATR-ZigZag major swings, and zone state to stabilise regardless of how
    # much history the caller feeds us (live loop feeds 60k M1 bars = 6 weeks).
    warmup = 3000
    state: dict = {}

    progress(f"Scanning {n:,} M1 bars for setups ...")
    report_every = max(n // 20, 1)

    for i in range(n):
        row = df.iloc[i]
        try:
            sig = _evaluate_row(i, row, cfg, state)
        except Exception as e:
            progress(f"  row {i} error: {e}")
            sig = None
        if i < warmup:
            continue
        if sig is not None:
            # BOS_CONT one-shot arm: mirror live trader — after a fill (in
            # scan, signal emit == fill since there's no broker-rejection
            # simulation), arm the key so subsequent consecutive BOS bars in
            # the same leg don't pyramid additional positions. Reset by
            # opposite CHoCH in Phase 1b of _evaluate_row (signals.py).
            # v0.9.13 live already does this in LiveTrader._handle_signal();
            # scan()/backtest missed it, letting pyramid storms like the
            # v0.9.7 08:35→08:39 5x SHORT re-appear in batch backtests.
            if sig.setup_kind == "BOS_CONT":
                state[f"_bos_entered_{sig.direction:+d}"] = True
            signals.append(sig)
        if i % report_every == 0:
            progress(f"  ... bar {i:,}/{n:,}  signals found: {len(signals)}")

    progress(f"Scan complete: {len(signals):,} signals.")

    # Dedupe: same direction within 5 M1 bars = same setup, keep higher grade
    signals.sort(key=lambda s: s.time)
    grade_rank = {'A+': 4, 'A': 3, 'B': 2, 'C': 1}
    deduped: list[Signal] = []
    for s in signals:
        if deduped and (s.time - deduped[-1].time) < pd.Timedelta(minutes=5) \
                and s.direction == deduped[-1].direction:
            if grade_rank.get(s.grade, 0) > grade_rank.get(deduped[-1].grade, 0):
                deduped[-1] = s
            continue
        deduped.append(s)
    return deduped




def signals_to_frame(signals: list[Signal]) -> pd.DataFrame:
    """Convert a list of Signal objects to a DataFrame (for inspection/CSV)."""
    if not signals:
        return pd.DataFrame()
    rows = [s.to_dict() for s in signals]
    return pd.DataFrame(rows)
