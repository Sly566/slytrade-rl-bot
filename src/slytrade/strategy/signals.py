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

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

from .config import StrategyConfig, ConfluenceConfig, ExitPlan


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
    confluence: List[str] = field(default_factory=list)
    fails: List[str] = field(default_factory=list)
    # Debug
    trigger_tf: str = "M5"
    ob_tf: Optional[str] = None
    ob_top: Optional[float] = None
    ob_bottom: Optional[float] = None
    fvg_top: Optional[float] = None
    fvg_bottom: Optional[float] = None
    swing_target_tf: Optional[str] = None
    swing_target_price: Optional[float] = None
    atr_at_entry: float = 0.0
    htf_bias_summary: Dict[str, int] = field(default_factory=dict)
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


def _price_within_ob_bull(row, tf: str, close_col: str = 'close') -> Optional[Tuple[float, float]]:
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


def _price_within_ob_bear(row, tf: str, close_col: str = 'close') -> Optional[Tuple[float, float]]:
    top = row.get(f'{tf}_bear_ob_top', np.nan)
    bot = row.get(f'{tf}_bear_ob_bottom', np.nan)
    mit = row.get(f'{tf}_bear_ob_mitigated', True)
    c = row[close_col]
    if pd.isna(top) or pd.isna(bot) or bool(mit):
        return None
    if bot <= c <= top:
        return (float(top), float(bot))
    return None


def _price_touching_fvg_bull(row, tf: str, close_col: str = 'close') -> Optional[Tuple[float, float]]:
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


def _price_touching_fvg_bear(row, tf: str, close_col: str = 'close') -> Optional[Tuple[float, float]]:
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
           bonus_killzone: bool) -> Tuple[str, List[str]]:
    """Return (grade, list_of_confluence_tags). 'fail' if no grade met.

    Tags are only added for tiers that *pass*, so we don't pollute the list
    with partial-tier fragments (which caused duplicate TF tags when a higher
    tier checked a TF that a lower tier also checked).
    """
    tags: List[str] = []

    def _all_agree(tfs: Tuple[str, ...]) -> Tuple[bool, List[str]]:
        """Return (all_ok, [tag_for_each_agreeing_tf])."""
        ok_tags: List[str] = []
        for tf in tfs:
            b = row.get(f'{tf}_major_bias', 0)
            if pd.isna(b) or int(b) != direction:
                return (False, [])
            ok_tags.append(f'{tf}_bias_aligned')
        return (True, ok_tags)

    grade: Optional[str] = None
    extra: List[str] = []

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

def _killzone_tag(row, cfg_sess) -> Tuple[bool, str]:
    """Return (allowed, tag) for the current M1 bar's session."""
    in_lon_kz = bool(row.get('kz_london', False))
    in_ny_kz  = bool(row.get('kz_ny', False))
    in_as_kz  = bool(row.get('kz_asian', False))
    in_lon_o30 = bool(row.get('london_open_30', False))
    in_ny_o30  = bool(row.get('ny_open_30', False))
    in_off     = (row.get('session', '') == 'OFF')

    tags = []
    allowed = False
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
                   entry: float, risk: float) -> Tuple[Optional[str], Optional[float]]:
    """Runner target = nearest opposing major swing on a bias-aligned TF.

    We look at M15, H1, H4, D1 in that order (closest-liquidity first). The
    first one that is at least 2R away and within 8R becomes the runner target;
    if nothing qualifies we fall back to 3R (or 5R when multiple TFs align but
    are distant).
    """
    runner_min_r = 2.0
    runner_max_r = 8.0
    fallback_r = 3.0
    candidates = []
    for tf in ('M15', 'H1', 'H4', 'D1'):
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
                  ) -> Optional[Signal]:
    """Evaluate one M1 bar and return a Signal if all gates pass, else None.

    `state` is mutated across rows; it tracks the most recent unmitigated
    OBs/FVGs on each HTF, keyed by f"{tf}_{side}_{kind}" → dict(top, bot, bar).
    """
    c = float(row['close'])
    atr = float(row['atr_14']) if pd.notna(row['atr_14']) else 0.0
    if atr <= 0:
        return None

    # Gate 0: ATR sanity (skip dead/news-spike bars)
    atr_pct = atr / c if c > 0 else 0.0
    if atr_pct < cfg.confluence.min_atr_pct or atr_pct > cfg.confluence.max_atr_pct:
        return None

    # Gate 1: session / killzone
    allowed, kz_tag = _killzone_tag(row, cfg.sessions)
    if not allowed:
        return None

    # Update tracked zones from this row's HTF values. A zone becomes "active"
    # when it appears (top becomes notnull, mitigated=False) after a
    # displacement/BOS on the same TF. It stays active while mitigated==False,
    # and expires after `retest_window` M1 bars.
    trigger_tf = cfg.trigger_tf
    retest_window = 60  # M1 bars (~1 hour) to wait for a retest after displacement
    t_now = row['time']

    def _update_zone(tf: str, side: str, kind: str):
        """Track active zone if the current HTF row has an unmitigated one."""
        key = f"{tf}_{side}_{kind}"
        top_k = f"{tf}_{side}_{kind}_top"
        bot_k = f"{tf}_{side}_{kind}_bottom"
        mit_k = f"{tf}_{side}_{kind}_mitigated"
        top = row.get(top_k, np.nan)
        bot = row.get(bot_k, np.nan)
        mit = bool(row.get(mit_k, True))
        if pd.notna(top) and pd.notna(bot) and not mit:
            # Zone is active on this HTF bar; refresh state
            prev = state.get(key)
            if prev is None or (prev['top'] != top or prev['bot'] != bot):
                # New zone (or zone changed) — reset entry flag
                state[key] = {'top': float(top), 'bot': float(bot), 'set_at': t_now, 'fresh': True}
                state[f"{key}_entered"] = False
            else:
                # Same zone still active; just mark still-present
                prev['fresh'] = False
        elif key in state:
            # Zone mitigated or gone; expire and clear entry flag
            state[key]['mitigated'] = True
            state[f"{key}_entered"] = True

    # Track OBs and FVGs on all OB TFs (and on trigger TF)
    track_tfs = list(dict.fromkeys(list(cfg.confluence.ob_tfs) + [trigger_tf]))
    for tf in track_tfs:
        _update_zone(tf, 'bull', 'ob')
        _update_zone(tf, 'bear', 'ob')
        _update_zone(tf, 'bull', 'fvg')
        _update_zone(tf, 'bear', 'fvg')

    # Determine whether a fresh displacement/BOS just fired on trigger_tf
    bull_trig_fresh = (bool(row.get(f'{trigger_tf}_bull_disp', False)) or
                       bool(row.get(f'{trigger_tf}_minor_bos_up', False)) or
                       bool(row.get(f'{trigger_tf}_major_bos_up', False)) or
                       bool(row.get(f'{trigger_tf}_major_choch_up', False)))
    bear_trig_fresh = (bool(row.get(f'{trigger_tf}_bear_disp', False)) or
                       bool(row.get(f'{trigger_tf}_minor_bos_dn', False)) or
                       bool(row.get(f'{trigger_tf}_major_bos_dn', False)) or
                       bool(row.get(f'{trigger_tf}_major_choch_dn', False)))

    # For each direction, require that a trigger fired within the last
    # `retest_window` M1 bars. Track trigger timestamps.
    if bull_trig_fresh:
        state['_last_bull_trigger'] = t_now
    if bear_trig_fresh:
        state['_last_bear_trigger'] = t_now

    def _within_window(ts):
        if ts is None: return False
        return (t_now - ts) <= pd.Timedelta(minutes=retest_window)

    bull_window = _within_window(state.get('_last_bull_trigger'))
    bear_window = _within_window(state.get('_last_bear_trigger'))

    candidates: List[int] = []
    if bull_window and cfg.confluence.accept_longs: candidates.append(1)
    if bear_window and cfg.confluence.accept_shorts: candidates.append(-1)
    if not candidates:
        return None

    for direction in candidates:
        side = 'bull' if direction == 1 else 'bear'

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

        # Gate 4: zone kind / TF gating (battle-tested filter)
        if kind.upper() not in cfg.confluence.accept_zone_kinds:
            continue
        if ztf not in cfg.confluence.accept_ob_tfs:
            continue

        # Gate 5: HTF bias agreement + grade assignment
        grade, conf_tags = _grade(direction, row, cfg.confluence,
                                  bonus_killzone=('+' in kz_tag or 'open30' in kz_tag))
        if grade == 'fail':
            continue
        if grade not in cfg.confluence.accept_grades:
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

        # Gate 6: risk-in-ATR width filter (battle-tested: <1.2 ATR stops get hunted)
        risk_atr = risk / atr if atr > 0 else 0.0
        if risk_atr < cfg.confluence.min_risk_atr:
            continue
        if risk_atr > cfg.confluence.max_risk_atr:
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

def _strategy_columns(cfg: StrategyConfig) -> List[str]:
    """Return the list of columns actually needed by the strategy engine."""
    cols = ['time', 'open', 'high', 'low', 'close', 'atr_14', 'session',
            'kz_asian', 'kz_london', 'kz_ny', 'london_open_30', 'ny_open_30',
            'vol_spike',
            'bull_disp', 'bear_disp',
            'minor_bos_up', 'minor_bos_dn', 'minor_choch_up', 'minor_choch_dn',
            'major_bos_up', 'major_bos_dn', 'major_choch_up', 'major_choch_dn']
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
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load aligned M1 partitions into one frame (month-by-month concat)."""
    import pyarrow.parquet as pq
    from ..data.storage import discover_partitions
    base = root / f"symbol={symbol}" / "timeframe=M1"
    files = discover_partitions(base, "**/*.parquet")
    if not files:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
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
         cfg: Optional[StrategyConfig] = None,
         progress: Optional[Callable[[str], None]] = None) -> List[Signal]:
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

    signals: List[Signal] = []
    n = len(df)
    warmup = 500  # first 500 M1 bars (~8 hours) for indicator warmup
    state: Dict = {}

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
            signals.append(sig)
        if i % report_every == 0:
            progress(f"  ... bar {i:,}/{n:,}  signals found: {len(signals)}")

    progress(f"Scan complete: {len(signals):,} signals.")

    # Dedupe: same direction within 5 M1 bars = same setup, keep higher grade
    signals.sort(key=lambda s: s.time)
    grade_rank = {'A+': 4, 'A': 3, 'B': 2, 'C': 1}
    deduped: List[Signal] = []
    for s in signals:
        if deduped and (s.time - deduped[-1].time) < pd.Timedelta(minutes=5) \
                and s.direction == deduped[-1].direction:
            if grade_rank.get(s.grade, 0) > grade_rank.get(deduped[-1].grade, 0):
                deduped[-1] = s
            continue
        deduped.append(s)
    return deduped




def signals_to_frame(signals: List[Signal]) -> pd.DataFrame:
    """Convert a list of Signal objects to a DataFrame (for inspection/CSV)."""
    if not signals:
        return pd.DataFrame()
    rows = [s.to_dict() for s in signals]
    return pd.DataFrame(rows)
