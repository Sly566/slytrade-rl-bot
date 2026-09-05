"""SlyTrade Gymnasium Environment — wraps backtest engine as standard RL env.

The agent observes market state and takes actions (enter/exit/hold/sizing).
The environment walks through historical M1 bars using the same signal
pipeline and backtest engine as the live trader.

Observation Space (Box):
  - Market features: OHLC normalized, ATR, structure flags, zone proximity
  - Position state: direction, entry offset, P&L R-multiple, bars held
  - Account state: equity curve, current drawdown, win rate

Action Space (MultiDiscrete):
  - action_type: 0=HOLD, 1=CLOSE, 2=ENTER_LONG, 3=ENTER_SHORT
  - size_level: 0=min_lot(0.01), 1=working_lot(0.04), 2=2x_working(0.08)
  - sl_mult: 0=1.0x_ATR, 1=1.5x_ATR, 2=2.0x_ATR, 3=2.5x_ATR
  - tp_mult: 0=0.5R, 1=1.0R, 2=1.5R, 3=2.0R, 4=2.5R
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..backtest.specs import AccountSpec, SymbolSpec, spec_for_symbol
from ..data.features import DEFAULT_CONFIG, process_bars
from ..data.mtf_align import _asof_merge, _prep_htf_frame
from ..data.time import timeframe_timedelta
from ..strategy.config import StrategyConfig, rl_training_persona
from ..strategy.signals import _evaluate_row


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ============================================================
# Observation vector layout — comprehensive market state (98 features)
# ============================================================

# 0-3: Core price
_OBS_BID = 0
_OBS_ASK = 1
_OBS_ATR = 2
_OBS_SPREAD = 3

# 4-9: M1 Structure
_OBS_BULL_DISP = 4
_OBS_BEAR_DISP = 5
_OBS_BOS_UP = 6
_OBS_BOS_DN = 7
_OBS_CHOCH_UP = 8
_OBS_CHOCH_DN = 9

# 10-15: M5 Structure
_OBS_M5_BULL_DISP = 10
_OBS_M5_BEAR_DISP = 11
_OBS_M5_BOS_UP = 12
_OBS_M5_BOS_DN = 13
_OBS_M5_CHOCH_UP = 14
_OBS_M5_CHOCH_DN = 15

# 16-23: M15 Structure
_OBS_M15_BULL_DISP = 16
_OBS_M15_BEAR_DISP = 17
_OBS_M15_BOS_UP = 18
_OBS_M15_BOS_DN = 19
_OBS_M15_CHOCH_UP = 20
_OBS_M15_CHOCH_DN = 21
_OBS_M15_MAJOR_CHOCH_UP = 22
_OBS_M15_MAJOR_CHOCH_DN = 23

# 24-27: H1 Structure
_OBS_H1_BOS_UP = 24
_OBS_H1_BOS_DN = 25
_OBS_H1_CHOCH_UP = 26
_OBS_H1_CHOCH_DN = 27

# 28-31: H4 Structure
_OBS_H4_BOS_UP = 28
_OBS_H4_BOS_DN = 29
_OBS_H4_CHOCH_UP = 30
_OBS_H4_CHOCH_DN = 31

# 32-35: D1 Structure (daily trend)
_OBS_D1_BOS_UP = 32
_OBS_D1_BOS_DN = 33
_OBS_D1_CHOCH_UP = 34
_OBS_D1_CHOCH_DN = 35

# 36-39: W1 Structure (weekly trend)
_OBS_W1_BOS_UP = 36
_OBS_W1_BOS_DN = 37
_OBS_W1_CHOCH_UP = 38
_OBS_W1_CHOCH_DN = 39

# 40-42: Zone proximity
_OBS_OB_PROX = 40
_OBS_FVG_PROX = 41
_OBS_SWEEP_PROX = 42

# 43-48: Support/Resistance
_OBS_SR_SUPPORT_DIST = 43
_OBS_SR_RESISTANCE_DIST = 44
_OBS_SR_SUPPORT_COUNT = 45
_OBS_SR_RESISTANCE_COUNT = 46
_OBS_AT_SUPPORT = 47
_OBS_AT_RESISTANCE = 48

# 49-56: Supply/Demand + Premium/Discount
_OBS_IN_DEMAND_ZONE = 49
_OBS_IN_SUPPLY_ZONE = 50
_OBS_DEMAND_STRENGTH = 51
_OBS_SUPPLY_STRENGTH = 52
_OBS_DEMAND_DIST = 53
_OBS_SUPPLY_DIST = 54
_OBS_IN_PREMIUM = 55
_OBS_IN_DISCOUNT = 56

# 57-63: Position state
_OBS_POS_DIR = 57
_OBS_POS_R = 58
_OBS_POS_BARS = 59
_OBS_POS_AGE = 60
_OBS_POS_GRADE = 61
_OBS_POS_TRAIL_ACTIVE = 62
_OBS_POS_PARTIAL_CLOSED = 63

# 64-71: Account state
_OBS_EQUITY_CURVE = 64
_OBS_DRAWDOWN = 65
_OBS_WIN_RATE = 66
_OBS_EQUITY_MOMENTUM = 67
_OBS_CONSEC_WINS = 68
_OBS_CONSEC_LOSSES = 69

# 70-75: Killzone + Time
_OBS_KZ_ASIA = 70
_OBS_KZ_LONDON = 71
_OBS_KZ_NY = 72
_OBS_KZ_LONDON_NY_OVERLAP = 73
_OBS_HOUR_SIN = 74
_OBS_HOUR_COS = 75

# 76-77: Day of week
_OBS_DOW_SIN = 76
_OBS_DOW_COS = 77

# 78-80: ATR regime
_OBS_ATR_PCT_RANK = 78
_OBS_ATR_EXPANDING = 79
_OBS_ATR_CONTRACTING = 80

# 81-89: Tick microstructure
_OBS_TICK_BUY_RATIO = 81
_OBS_TICK_SELL_RATIO = 82
_OBS_TICK_SPREAD_MEAN = 83
_OBS_TICK_SPREAD_MAX = 84
_OBS_TICK_PRICE_VELOCITY = 85
_OBS_TICK_VOLUME_IMBALANCE = 86
_OBS_TICK_ABSORPTION = 87
_OBS_TICK_LARGE_TRADE = 88
_OBS_TICK_COUNT = 89

# 90-93: News
_OBS_NEWS_MINUTES_TO = 90
_OBS_NEWS_MINUTES_SINCE = 91
_OBS_NEWS_IN_WINDOW = 92
_OBS_NEWS_IMPACT_SCORE = 93

# 94-95: Volume
_OBS_VOL_RATIO = 94
_OBS_VOL_SPIKE = 95

# 96-97: Liquidity sweeps
_OBS_BULL_SWEEP = 96
_OBS_BEAR_SWEEP = 97

OBS_DIM = 98

# Action space
ACT_HOLD = 0
ACT_CLOSE = 1
ACT_ENTER_LONG = 2
ACT_ENTER_SHORT = 3

# Size levels
SIZE_MIN = 0       # 0.01 lots
SIZE_WORKING = 1   # 0.04 lots
SIZE_DOUBLE = 2    # 0.08 lots

# SL/TP multipliers
SL_MULTS = [1.0, 1.5, 2.0, 2.5]   # × ATR
TP_MULTS = [0.5, 1.0, 1.5, 2.0, 2.5]  # × R


def _safe_from_row(
    row: Any,
    pos_dir: int = 0,
    pos_entry: float = 0.0,
    pos_risk: float = 0.0,
    pos_bars: int = 0,
    equity: float = 2000.0,
    peak_equity: float = 2000.0,
    starting_equity: float = 2000.0,
    recent_wins: list[bool] | None = None,
    time_stop_bars: int = 60,
    pos_grade: str = "",
    pos_trail_active: bool = False,
    pos_partial_closed: bool = False,
    equity_history: list[float] | None = None,
) -> np.ndarray:
    """Build observation vector from a bar row (dict or Series).

    This is the LIVE/BACKTEST version of env._get_obs() — reads from
    aligned DataFrame rows instead of pre-extracted numpy arrays.
    Used by serve.py for real-time inference.
    """
    def _g(key, default=0.0):
        try:
            v = row.get(key, default) if hasattr(row, 'get') else getattr(row, key, default)
            return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default
        except Exception:
            return default

    def _b(key):
        return 1.0 if _g(key) else 0.0

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    price = _g("close")
    atr = max(_g("atr_14", 0.001), 0.001)

    # Core price
    price_norm = price / 1000.0 if price > 0 else 0.0
    obs[_OBS_BID] = price_norm
    obs[_OBS_ASK] = price_norm + 0.0002
    obs[_OBS_ATR] = atr / max(price, 1.0)
    actual_spread = _g("tick_spread_mean")
    obs[_OBS_SPREAD] = (actual_spread / atr) if actual_spread > 0 else (0.2 / max(atr, 0.001))

    # M1 structure
    obs[_OBS_BULL_DISP] = _b("bull_disp"); obs[_OBS_BEAR_DISP] = _b("bear_disp")
    obs[_OBS_BOS_UP] = _b("minor_bos_up"); obs[_OBS_BOS_DN] = _b("minor_bos_dn")
    obs[_OBS_CHOCH_UP] = _b("minor_choch_up"); obs[_OBS_CHOCH_DN] = _b("minor_choch_dn")
    # M5
    obs[_OBS_M5_BULL_DISP] = _b("M5_bull_disp"); obs[_OBS_M5_BEAR_DISP] = _b("M5_bear_disp")
    obs[_OBS_M5_BOS_UP] = _b("M5_minor_bos_up"); obs[_OBS_M5_BOS_DN] = _b("M5_minor_bos_dn")
    obs[_OBS_M5_CHOCH_UP] = _b("M5_minor_choch_up"); obs[_OBS_M5_CHOCH_DN] = _b("M5_minor_choch_dn")
    # M15
    obs[_OBS_M15_BULL_DISP] = _b("M15_bull_disp"); obs[_OBS_M15_BEAR_DISP] = _b("M15_bear_disp")
    obs[_OBS_M15_BOS_UP] = _b("M15_minor_bos_up"); obs[_OBS_M15_BOS_DN] = _b("M15_minor_bos_dn")
    obs[_OBS_M15_CHOCH_UP] = _b("M15_minor_choch_up"); obs[_OBS_M15_CHOCH_DN] = _b("M15_minor_choch_dn")
    obs[_OBS_M15_MAJOR_CHOCH_UP] = _b("M15_major_choch_up"); obs[_OBS_M15_MAJOR_CHOCH_DN] = _b("M15_major_choch_dn")
    # HTF
    obs[_OBS_H1_BOS_UP] = _b("H1_minor_bos_up"); obs[_OBS_H1_BOS_DN] = _b("H1_minor_bos_dn")
    obs[_OBS_H1_CHOCH_UP] = _b("H1_minor_choch_up"); obs[_OBS_H1_CHOCH_DN] = _b("H1_minor_choch_dn")
    obs[_OBS_H4_BOS_UP] = _b("H4_minor_bos_up"); obs[_OBS_H4_BOS_DN] = _b("H4_minor_bos_dn")
    obs[_OBS_H4_CHOCH_UP] = _b("H4_minor_choch_up"); obs[_OBS_H4_CHOCH_DN] = _b("H4_minor_choch_dn")
    # D1
    obs[_OBS_D1_BOS_UP] = _b("D1_minor_bos_up"); obs[_OBS_D1_BOS_DN] = _b("D1_minor_bos_dn")
    obs[_OBS_D1_CHOCH_UP] = _b("D1_minor_choch_up"); obs[_OBS_D1_CHOCH_DN] = _b("D1_minor_choch_dn")
    # W1
    obs[_OBS_W1_BOS_UP] = _b("W1_minor_bos_up"); obs[_OBS_W1_BOS_DN] = _b("W1_minor_bos_dn")
    obs[_OBS_W1_CHOCH_UP] = _b("W1_minor_choch_up"); obs[_OBS_W1_CHOCH_DN] = _b("W1_minor_choch_dn")
    # Zone proximity
    obs[_OBS_OB_PROX] = min(_g("ob_proximity", 10.0), 10.0)
    obs[_OBS_FVG_PROX] = min(_g("fvg_proximity", 10.0), 10.0)
    obs[_OBS_SWEEP_PROX] = min(_g("sweep_proximity", 10.0), 10.0)
    # S/R
    obs[_OBS_SR_SUPPORT_DIST] = min(_g("sr_support_dist", 10.0), 10.0)
    obs[_OBS_SR_RESISTANCE_DIST] = min(_g("sr_resistance_dist", 10.0), 10.0)
    obs[_OBS_SR_SUPPORT_COUNT] = min(_g("sr_support_count"), 5.0) / 5.0
    obs[_OBS_SR_RESISTANCE_COUNT] = min(_g("sr_resistance_count"), 5.0) / 5.0
    obs[_OBS_AT_SUPPORT] = _b("at_support"); obs[_OBS_AT_RESISTANCE] = _b("at_resistance")
    # S/D
    obs[_OBS_IN_DEMAND_ZONE] = _b("in_demand_zone"); obs[_OBS_IN_SUPPLY_ZONE] = _b("in_supply_zone")
    obs[_OBS_DEMAND_STRENGTH] = _g("demand_zone_strength") / 3.0
    obs[_OBS_SUPPLY_STRENGTH] = _g("supply_zone_strength") / 3.0
    obs[_OBS_DEMAND_DIST] = min(_g("demand_zone_dist", 10.0), 10.0)
    obs[_OBS_SUPPLY_DIST] = min(_g("supply_zone_dist", 10.0), 10.0)
    # Premium/Discount
    obs[_OBS_IN_PREMIUM] = _b("in_premium"); obs[_OBS_IN_DISCOUNT] = _b("in_discount")
    # Position state
    obs[_OBS_POS_DIR] = float(pos_dir)
    if pos_dir != 0 and pos_risk > 0:
        r_dist = (price - pos_entry) if pos_dir == 1 else (pos_entry - price)
        obs[_OBS_POS_R] = r_dist / pos_risk
    obs[_OBS_POS_BARS] = pos_bars / max(time_stop_bars, 1)
    obs[_OBS_POS_AGE] = pos_bars / max(time_stop_bars, 1)
    grade_map = {"A+": 4, "A": 3, "B": 2, "C": 1}
    obs[_OBS_POS_GRADE] = grade_map.get(pos_grade, 0) / 4.0
    obs[_OBS_POS_TRAIL_ACTIVE] = 1.0 if pos_trail_active else 0.0
    obs[_OBS_POS_PARTIAL_CLOSED] = 1.0 if pos_partial_closed else 0.0
    # Account
    obs[_OBS_EQUITY_CURVE] = equity / max(starting_equity, 1.0)
    obs[_OBS_DRAWDOWN] = (peak_equity - equity) / max(peak_equity, 1.0)
    if recent_wins:
        obs[_OBS_WIN_RATE] = sum(recent_wins[-20:]) / len(recent_wins[-20:])
    if equity_history and len(equity_history) > 100:
        obs[_OBS_EQUITY_MOMENTUM] = (equity_history[-1] - equity_history[-100]) / max(equity_history[-100], 1.0)
    # Killzone + time
    try:
        ts = pd.Timestamp(row["time"] if hasattr(row, '__getitem__') else row.time)
        hour = ts.hour; dow = ts.dayofweek
        obs[_OBS_KZ_ASIA] = 1.0 if 0 <= hour < 8 else 0.0
        obs[_OBS_KZ_LONDON] = 1.0 if 7 <= hour < 16 else 0.0
        obs[_OBS_KZ_NY] = 1.0 if 12 <= hour < 21 else 0.0
        obs[_OBS_KZ_LONDON_NY_OVERLAP] = 1.0 if 12 <= hour < 16 else 0.0
        obs[_OBS_HOUR_SIN] = np.sin(2 * np.pi * hour / 24.0)
        obs[_OBS_HOUR_COS] = np.cos(2 * np.pi * hour / 24.0)
        obs[_OBS_DOW_SIN] = np.sin(2 * np.pi * dow / 7.0)
        obs[_OBS_DOW_COS] = np.cos(2 * np.pi * dow / 7.0)
    except Exception:
        pass
    # ATR regime
    obs[_OBS_ATR_PCT_RANK] = _g("atr_pct_rank", 0.5)
    obs[_OBS_ATR_EXPANDING] = _b("atr_expanding")
    obs[_OBS_ATR_CONTRACTING] = _b("atr_contracting")
    # Tick microstructure
    obs[_OBS_TICK_BUY_RATIO] = _g("tick_buy_ratio"); obs[_OBS_TICK_SELL_RATIO] = _g("tick_sell_ratio")
    obs[_OBS_TICK_SPREAD_MEAN] = _g("tick_spread_mean"); obs[_OBS_TICK_SPREAD_MAX] = _g("tick_spread_max")
    obs[_OBS_TICK_PRICE_VELOCITY] = _g("tick_price_velocity")
    obs[_OBS_TICK_VOLUME_IMBALANCE] = _g("tick_volume_imbalance")
    obs[_OBS_TICK_ABSORPTION] = _g("tick_absorption")
    obs[_OBS_TICK_LARGE_TRADE] = _g("tick_large_trade_ratio")
    obs[_OBS_TICK_COUNT] = min(_g("tick_count") / 1000.0, 1.0)
    # News
    obs[_OBS_NEWS_MINUTES_TO] = min(_g("minutes_to_next_high", 999.0), 999.0) / 999.0
    obs[_OBS_NEWS_MINUTES_SINCE] = min(_g("minutes_since_last_high", 999.0), 999.0) / 999.0
    obs[_OBS_NEWS_IN_WINDOW] = _b("in_news_window")
    obs[_OBS_NEWS_IMPACT_SCORE] = _g("news_impact_score") / 3.0
    # Volume
    obs[_OBS_VOL_RATIO] = _g("tick_vol_ratio", 1.0)
    obs[_OBS_VOL_SPIKE] = _b("vol_spike")
    # Liquidity sweeps
    obs[_OBS_BULL_SWEEP] = _b("bull_liq_sweep"); obs[_OBS_BEAR_SWEEP] = _b("bear_liq_sweep")

    np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return obs


class SlyTradeEnv(gym.Env):
    """Gymnasium environment wrapping the SlyTrade backtest engine.

    The agent walks through historical M1 bars, observing market state
    and taking actions (enter/exit/hold). The environment computes P&L
    using the same signal pipeline as the live trader.

    Args:
        aligned_df: Pre-aligned M1 bars with HTF features (from backtest data)
        cfg: Strategy configuration (default: rl_training_persona)
        spec: Symbol specification
        acct: Account specification
        max_bars: Maximum bars per episode (default: all bars)
        time_stop_bars: Time-stop in bars (default: 60)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        aligned_df: pd.DataFrame,
        cfg: StrategyConfig | None = None,
        spec: SymbolSpec | None = None,
        acct: AccountSpec | None = None,
        *,
        max_bars: int | None = None,
        time_stop_bars: int = 60,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Data — convert to numpy for fast O(1) access (100x faster than pandas iloc)
        aligned_df = aligned_df.reset_index(drop=True)
        self.n_bars = min(max_bars or len(aligned_df), len(aligned_df))

        # Pre-extract columns as numpy arrays
        _COL_MAP = {
            # Core price
            "close": "close", "atr_14": "atr_14", "time": "time",
            # M1 structure
            "bull_disp": "bull_disp", "bear_disp": "bear_disp",
            "minor_bos_up": "minor_bos_up", "minor_bos_dn": "minor_bos_dn",
            "minor_choch_up": "minor_choch_up", "minor_choch_dn": "minor_choch_dn",
            # M5 structure
            "M5_bull_disp": "M5_bull_disp", "M5_bear_disp": "M5_bear_disp",
            "M5_minor_bos_up": "M5_minor_bos_up", "M5_minor_bos_dn": "M5_minor_bos_dn",
            "M5_minor_choch_up": "M5_minor_choch_up", "M5_minor_choch_dn": "M5_minor_choch_dn",
            # M15 structure
            "M15_bull_disp": "M15_bull_disp", "M15_bear_disp": "M15_bear_disp",
            "M15_minor_bos_up": "M15_minor_bos_up", "M15_minor_bos_dn": "M15_minor_bos_dn",
            "M15_minor_choch_up": "M15_minor_choch_up", "M15_minor_choch_dn": "M15_minor_choch_dn",
            "M15_major_choch_up": "M15_major_choch_up", "M15_major_choch_dn": "M15_major_choch_dn",
            # HTF structure (Gap 6)
            "H1_minor_bos_up": "H1_minor_bos_up", "H1_minor_bos_dn": "H1_minor_bos_dn",
            "H1_minor_choch_up": "H1_minor_choch_up", "H1_minor_choch_dn": "H1_minor_choch_dn",
            "H4_minor_bos_up": "H4_minor_bos_up", "H4_minor_bos_dn": "H4_minor_bos_dn",
            "H4_minor_choch_up": "H4_minor_choch_up", "H4_minor_choch_dn": "H4_minor_choch_dn",
            # D1 structure
            "D1_minor_bos_up": "D1_minor_bos_up", "D1_minor_bos_dn": "D1_minor_bos_dn",
            "D1_minor_choch_up": "D1_minor_choch_up", "D1_minor_choch_dn": "D1_minor_choch_dn",
            # W1 structure
            "W1_minor_bos_up": "W1_minor_bos_up", "W1_minor_bos_dn": "W1_minor_bos_dn",
            "W1_minor_choch_up": "W1_minor_choch_up", "W1_minor_choch_dn": "W1_minor_choch_dn",
            # Zone proximity
            "ob_proximity": "ob_proximity", "fvg_proximity": "fvg_proximity",
            "sweep_proximity": "sweep_proximity",
            # S/R zones
            "sr_support_dist": "sr_support_dist", "sr_resistance_dist": "sr_resistance_dist",
            "sr_support_count": "sr_support_count", "sr_resistance_count": "sr_resistance_count",
            "at_support": "at_support", "at_resistance": "at_resistance",
            # Supply/Demand zones
            "in_demand_zone": "in_demand_zone", "in_supply_zone": "in_supply_zone",
            "demand_zone_strength": "demand_zone_strength", "supply_zone_strength": "supply_zone_strength",
            "demand_zone_dist": "demand_zone_dist", "supply_zone_dist": "supply_zone_dist",
            # Premium/Discount
            "in_premium": "in_premium", "in_discount": "in_discount",
            # ATR regime
            "atr_pct_rank": "atr_pct_rank", "atr_expanding": "atr_expanding",
            "atr_contracting": "atr_contracting",
            # Volume
            "tick_vol_ratio": "tick_vol_ratio", "vol_spike": "vol_spike",
            # Liquidity sweeps
            "bull_liq_sweep": "bull_liq_sweep", "bear_liq_sweep": "bear_liq_sweep",
            # Tick microstructure
            "tick_buy_ratio": "tick_buy_ratio", "tick_sell_ratio": "tick_sell_ratio",
            "tick_spread_mean": "tick_spread_mean", "tick_spread_max": "tick_spread_max",
            "tick_price_velocity": "tick_price_velocity", "tick_volume_imbalance": "tick_volume_imbalance",
            "tick_absorption": "tick_absorption", "tick_large_trade_ratio": "tick_large_trade_ratio",
            "tick_count": "tick_count",
            # News
            "minutes_to_next_high": "minutes_to_next_high",
            "minutes_since_last_high": "minutes_since_last_high",
            "in_news_window": "in_news_window", "news_impact_score": "news_impact_score",
        }
        self._col_arrays = {}
        for key, col in _COL_MAP.items():
            if col in aligned_df.columns:
                if col == "time":
                    self._col_arrays[key] = aligned_df[col].values  # keep as datetime64
                elif aligned_df[col].dtype == bool:
                    self._col_arrays[key] = aligned_df[col].values.astype(np.float32)
                else:
                    self._col_arrays[key] = pd.to_numeric(aligned_df[col], errors="coerce").fillna(0.0).values.astype(np.float64)
            else:
                self._col_arrays[key] = np.zeros(len(aligned_df), dtype=np.float64)

        # Keep aligned for _evaluate_row compatibility (lightweight reference)
        self.aligned = aligned_df

        # Config
        self.cfg = cfg or rl_training_persona()
        self.spec = spec or spec_for_symbol("XAUUSDm")
        self.acct = acct or AccountSpec(
            starting_equity=2000.0, currency="ZAR",
            leverage=2000, fx_to_account={"USD": 18.5},
        )
        self.time_stop_bars = time_stop_bars

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        # MultiDiscrete: [action_type, size_level, sl_mult, tp_mult]
        self.action_space = spaces.MultiDiscrete([4, 3, 4, 5])

        # State
        self._reset_state()

    def _reset_state(self):
        """Reset internal state for a new episode."""
        self._bar_idx = 0
        self._state: dict = {}
        self._equity = self.acct.starting_equity
        self._peak_equity = self._equity
        self._starting_equity = self._equity

        # Position state (Gap 7: expanded)
        self._pos_dir = 0
        self._pos_entry = 0.0
        self._pos_sl = 0.0
        self._pos_tp = 0.0
        self._pos_lots = 0.0
        self._pos_bars = 0
        self._pos_risk_per_unit = 0.0
        self._pos_grade = ""         # A+, A, B, C
        self._pos_trail_active = False
        self._pos_partial_closed = False
        self._pos_original_lots = 0.0
        self._pos_best_price = 0.0   # for trailing

        # Hybrid ladder state (Gap 2: match live)
        self._tp1_hit = False
        self._tp2_hit = False
        self._runner_active = False

        # Trade history
        self._trades: list[dict] = []
        self._recent_wins: list[bool] = []
        self._equity_history: list[float] = [self._equity]  # Gap 10

    def _safe(self, key: str, i: int, default: float = 0.0) -> float:
        """Safely read a value from _col_arrays."""
        arr = self._col_arrays.get(key)
        if arr is not None and i < len(arr):
            v = arr[i]
            return float(v) if np.isfinite(v) else default
        return default

    def _safe_bool(self, key: str, i: int) -> float:
        """Safely read a bool value from _col_arrays."""
        v = self._safe(key, i, 0.0)
        return 1.0 if v else 0.0

    def _get_obs(self) -> np.ndarray:
        """Build comprehensive observation vector from current bar and position state.
        90 features covering: price, structure (M1-M15-M30-H1-H4), zones,
        S/R, S/D, position state, account state, killzones, time, ATR regime,
        tick microstructure, news, and volume.
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        i = self._bar_idx
        if i >= self.n_bars:
            return obs

        ca = self._col_arrays
        s = self._safe
        sb = self._safe_bool
        price = ca["close"][i]
        atr = ca["atr_14"][i]
        if not np.isfinite(atr) or atr <= 0:
            atr = 0.001

        # === Core price ===
        price_norm = price / 1000.0 if price > 0 else 0.0
        obs[_OBS_BID] = price_norm
        obs[_OBS_ASK] = price_norm + 0.0002
        obs[_OBS_ATR] = atr / max(price, 1.0)
        # Gap 9: Real spread from tick data instead of hardcoded 0.2
        actual_spread = s("tick_spread_mean", i)
        if actual_spread > 0:
            obs[_OBS_SPREAD] = actual_spread / atr
        else:
            obs[_OBS_SPREAD] = 0.2 / max(atr, 0.001)  # fallback

        # === M1 Structure ===
        obs[_OBS_BULL_DISP] = sb("bull_disp", i)
        obs[_OBS_BEAR_DISP] = sb("bear_disp", i)
        obs[_OBS_BOS_UP] = sb("minor_bos_up", i)
        obs[_OBS_BOS_DN] = sb("minor_bos_dn", i)
        obs[_OBS_CHOCH_UP] = sb("minor_choch_up", i)
        obs[_OBS_CHOCH_DN] = sb("minor_choch_dn", i)

        # === M5 Structure ===
        obs[_OBS_M5_BULL_DISP] = sb("M5_bull_disp", i)
        obs[_OBS_M5_BEAR_DISP] = sb("M5_bear_disp", i)
        obs[_OBS_M5_BOS_UP] = sb("M5_minor_bos_up", i)
        obs[_OBS_M5_BOS_DN] = sb("M5_minor_bos_dn", i)
        obs[_OBS_M5_CHOCH_UP] = sb("M5_minor_choch_up", i)
        obs[_OBS_M5_CHOCH_DN] = sb("M5_minor_choch_dn", i)

        # === M15 Structure ===
        obs[_OBS_M15_BULL_DISP] = sb("M15_bull_disp", i)
        obs[_OBS_M15_BEAR_DISP] = sb("M15_bear_disp", i)
        obs[_OBS_M15_BOS_UP] = sb("M15_minor_bos_up", i)
        obs[_OBS_M15_BOS_DN] = sb("M15_minor_bos_dn", i)
        obs[_OBS_M15_CHOCH_UP] = sb("M15_minor_choch_up", i)
        obs[_OBS_M15_CHOCH_DN] = sb("M15_minor_choch_dn", i)
        obs[_OBS_M15_MAJOR_CHOCH_UP] = sb("M15_major_choch_up", i)
        obs[_OBS_M15_MAJOR_CHOCH_DN] = sb("M15_major_choch_dn", i)

        # === HTF Structure (Gap 6) ===
        obs[_OBS_H1_BOS_UP] = sb("H1_minor_bos_up", i)
        obs[_OBS_H1_BOS_DN] = sb("H1_minor_bos_dn", i)
        obs[_OBS_H1_CHOCH_UP] = sb("H1_minor_choch_up", i)
        obs[_OBS_H1_CHOCH_DN] = sb("H1_minor_choch_dn", i)
        obs[_OBS_H4_BOS_UP] = sb("H4_minor_bos_up", i)
        obs[_OBS_H4_BOS_DN] = sb("H4_minor_bos_dn", i)
        obs[_OBS_H4_CHOCH_UP] = sb("H4_minor_choch_up", i)
        obs[_OBS_H4_CHOCH_DN] = sb("H4_minor_choch_dn", i)

        # === D1 Structure (daily trend) ===
        obs[_OBS_D1_BOS_UP] = sb("D1_minor_bos_up", i)
        obs[_OBS_D1_BOS_DN] = sb("D1_minor_bos_dn", i)
        obs[_OBS_D1_CHOCH_UP] = sb("D1_minor_choch_up", i)
        obs[_OBS_D1_CHOCH_DN] = sb("D1_minor_choch_dn", i)

        # === W1 Structure (weekly trend) ===
        obs[_OBS_W1_BOS_UP] = sb("W1_minor_bos_up", i)
        obs[_OBS_W1_BOS_DN] = sb("W1_minor_bos_dn", i)
        obs[_OBS_W1_CHOCH_UP] = sb("W1_minor_choch_up", i)
        obs[_OBS_W1_CHOCH_DN] = sb("W1_minor_choch_dn", i)

        # === Zone proximity ===
        obs[_OBS_OB_PROX] = min(s("ob_proximity", i, 10.0), 10.0)
        obs[_OBS_FVG_PROX] = min(s("fvg_proximity", i, 10.0), 10.0)
        obs[_OBS_SWEEP_PROX] = min(s("sweep_proximity", i, 10.0), 10.0)

        # === S/R zones ===
        obs[_OBS_SR_SUPPORT_DIST] = min(s("sr_support_dist", i, 10.0), 10.0)
        obs[_OBS_SR_RESISTANCE_DIST] = min(s("sr_resistance_dist", i, 10.0), 10.0)
        obs[_OBS_SR_SUPPORT_COUNT] = min(s("sr_support_count", i), 5.0) / 5.0
        obs[_OBS_SR_RESISTANCE_COUNT] = min(s("sr_resistance_count", i), 5.0) / 5.0
        obs[_OBS_AT_SUPPORT] = sb("at_support", i)
        obs[_OBS_AT_RESISTANCE] = sb("at_resistance", i)

        # === Supply/Demand zones ===
        obs[_OBS_IN_DEMAND_ZONE] = sb("in_demand_zone", i)
        obs[_OBS_IN_SUPPLY_ZONE] = sb("in_supply_zone", i)
        obs[_OBS_DEMAND_STRENGTH] = s("demand_zone_strength", i) / 3.0
        obs[_OBS_SUPPLY_STRENGTH] = s("supply_zone_strength", i) / 3.0
        obs[_OBS_DEMAND_DIST] = min(s("demand_zone_dist", i, 10.0), 10.0)
        obs[_OBS_SUPPLY_DIST] = min(s("supply_zone_dist", i, 10.0), 10.0)

        # === Premium/Discount ===
        obs[_OBS_IN_PREMIUM] = sb("in_premium", i)
        obs[_OBS_IN_DISCOUNT] = sb("in_discount", i)

        # === Position state (Gap 7: expanded) ===
        obs[_OBS_POS_DIR] = float(self._pos_dir)
        if self._pos_dir != 0 and self._pos_risk_per_unit > 0:
            r_dist = (price - self._pos_entry) if self._pos_dir == 1 else (self._pos_entry - price)
            obs[_OBS_POS_R] = r_dist / self._pos_risk_per_unit
        obs[_OBS_POS_BARS] = self._pos_bars / max(self.time_stop_bars, 1)
        obs[_OBS_POS_AGE] = self._pos_bars / max(self.time_stop_bars, 1)
        # Grade encoding: A+=4, A=3, B=2, C=1, none=0
        grade_map = {"A+": 4, "A": 3, "B": 2, "C": 1}
        obs[_OBS_POS_GRADE] = grade_map.get(self._pos_grade, 0) / 4.0
        obs[_OBS_POS_TRAIL_ACTIVE] = 1.0 if self._pos_trail_active else 0.0
        obs[_OBS_POS_PARTIAL_CLOSED] = 1.0 if self._pos_partial_closed else 0.0

        # === Account state (Gap 10: expanded) ===
        obs[_OBS_EQUITY_CURVE] = self._equity / max(self._starting_equity, 1.0)
        dd = (self._peak_equity - self._equity) / max(self._peak_equity, 1.0)
        obs[_OBS_DRAWDOWN] = dd
        if self._recent_wins:
            obs[_OBS_WIN_RATE] = sum(self._recent_wins[-20:]) / len(self._recent_wins[-20:])
        # Equity momentum: change over last 100 bars
        if len(self._equity_history) > 100:
            eq_now = self._equity_history[-1]
            eq_100 = self._equity_history[-100]
            obs[_OBS_EQUITY_MOMENTUM] = (eq_now - eq_100) / max(eq_100, 1.0)
        # Consecutive wins/losses
        if self._recent_wins:
            consec_w = 0
            consec_l = 0
            for w in reversed(self._recent_wins):
                if w:
                    consec_w += 1
                else:
                    break
            for w in reversed(self._recent_wins):
                if not w:
                    consec_l += 1
                else:
                    break
            obs[_OBS_CONSEC_WINS] = min(consec_w, 10) / 10.0
            obs[_OBS_CONSEC_LOSSES] = min(consec_l, 10) / 10.0

        # === Killzone ===
        try:
            ts_val = ca["time"][i]
            ts = pd.Timestamp(ts_val)
            hour = ts.hour
            dow = ts.dayofweek
            obs[_OBS_KZ_ASIA] = 1.0 if 0 <= hour < 8 else 0.0
            obs[_OBS_KZ_LONDON] = 1.0 if 7 <= hour < 16 else 0.0
            obs[_OBS_KZ_NY] = 1.0 if 12 <= hour < 21 else 0.0
            obs[_OBS_KZ_LONDON_NY_OVERLAP] = 1.0 if 12 <= hour < 16 else 0.0
            obs[_OBS_HOUR_SIN] = np.sin(2 * np.pi * hour / 24.0)
            obs[_OBS_HOUR_COS] = np.cos(2 * np.pi * hour / 24.0)
            obs[_OBS_DOW_SIN] = np.sin(2 * np.pi * dow / 7.0)
            obs[_OBS_DOW_COS] = np.cos(2 * np.pi * dow / 7.0)
        except Exception:
            pass

        # === ATR regime ===
        obs[_OBS_ATR_PCT_RANK] = s("atr_pct_rank", i, 0.5)
        obs[_OBS_ATR_EXPANDING] = sb("atr_expanding", i)
        obs[_OBS_ATR_CONTRACTING] = sb("atr_contracting", i)

        # === Tick microstructure ===
        obs[_OBS_TICK_BUY_RATIO] = s("tick_buy_ratio", i)
        obs[_OBS_TICK_SELL_RATIO] = s("tick_sell_ratio", i)
        obs[_OBS_TICK_SPREAD_MEAN] = s("tick_spread_mean", i)
        obs[_OBS_TICK_SPREAD_MAX] = s("tick_spread_max", i)
        obs[_OBS_TICK_PRICE_VELOCITY] = s("tick_price_velocity", i)
        obs[_OBS_TICK_VOLUME_IMBALANCE] = s("tick_volume_imbalance", i)
        obs[_OBS_TICK_ABSORPTION] = s("tick_absorption", i)
        obs[_OBS_TICK_LARGE_TRADE] = s("tick_large_trade_ratio", i)
        tick_ct = s("tick_count", i)
        obs[_OBS_TICK_COUNT] = min(tick_ct / 1000.0, 1.0)

        # === News features (Gap 5) ===
        obs[_OBS_NEWS_MINUTES_TO] = min(s("minutes_to_next_high", i, 999.0), 999.0) / 999.0
        obs[_OBS_NEWS_MINUTES_SINCE] = min(s("minutes_since_last_high", i, 999.0), 999.0) / 999.0
        obs[_OBS_NEWS_IN_WINDOW] = sb("in_news_window", i)
        obs[_OBS_NEWS_IMPACT_SCORE] = s("news_impact_score", i) / 3.0

        # === Volume ===
        obs[_OBS_VOL_RATIO] = s("tick_vol_ratio", i, 1.0)
        obs[_OBS_VOL_SPIKE] = sb("vol_spike", i)

        # === Liquidity sweeps ===
        obs[_OBS_BULL_SWEEP] = sb("bull_liq_sweep", i)
        obs[_OBS_BEAR_SWEEP] = sb("bear_liq_sweep", i)

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _get_info(self) -> dict:
        """Return info dict for debugging/logging."""
        return {
            "bar_idx": self._bar_idx,
            "equity": self._equity,
            "drawdown": (self._peak_equity - self._equity) / max(self._peak_equity, 1.0),
            "pos_dir": self._pos_dir,
            "pos_bars": self._pos_bars,
            "n_trades": len(self._trades),
            "win_rate": sum(self._recent_wins[-20:]) / max(len(self._recent_wins[-20:]), 1),
        }

    def reset(self, *, seed=None, options=None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one step: process action, advance one bar, compute reward.

        Args:
            action: MultiDiscrete array [action_type, size_level, sl_mult, tp_mult]

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Handle both torch tensors and numpy arrays
        action = np.asarray(action).flatten()
        action_type = int(action[0])
        size_level = int(action[1])
        sl_mult_idx = int(action[2])
        tp_mult_idx = int(action[3])

        i = self._bar_idx
        ca = self._col_arrays
        price = ca["close"][i]
        atr = ca["atr_14"][i]
        if not np.isfinite(atr) or atr <= 0:
            atr = 0.0
        bid = price
        ask = price + 0.0002

        reward = 0.0
        trade_closed = False
        close_pnl = 0.0

        # --- Process action ---
        if action_type == ACT_CLOSE and self._pos_dir != 0:
            # Close position
            if self._pos_dir == 1:
                close_pnl = (bid - self._pos_entry) * self._pos_lots * self.spec.contract_size
            else:
                close_pnl = (self._pos_entry - ask) * self._pos_lots * self.spec.contract_size
            # Convert to account currency
            close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
            self._equity += close_pnl_acct
            self._peak_equity = max(self._peak_equity, self._equity)
            self._recent_wins.append(close_pnl_acct > 0)
            self._trades.append({
                "entry": self._pos_entry,
                "exit": price,
                "dir": self._pos_dir,
                "lots": self._pos_lots,
                "pnl": close_pnl_acct,
                "bars": self._pos_bars,
            })
            self._pos_dir = 0
            self._pos_entry = 0.0
            self._pos_sl = 0.0
            self._pos_tp = 0.0
            self._pos_lots = 0.0
            self._pos_bars = 0
            self._pos_risk_per_unit = 0.0
            trade_closed = True

        elif action_type == ACT_ENTER_LONG and self._pos_dir == 0:
            # Enter long
            lots = [0.01, 0.04, 0.08][size_level]
            sl_dist = SL_MULTS[sl_mult_idx] * max(atr, 0.01)
            tp_dist_mult = TP_MULTS[tp_mult_idx]
            self._pos_dir = 1
            self._pos_entry = ask
            self._pos_sl = ask - sl_dist
            self._pos_tp = ask + tp_dist_mult * sl_dist  # TP = tp_mult × R
            self._pos_lots = lots
            self._pos_bars = 0
            self._pos_risk_per_unit = sl_dist

        elif action_type == ACT_ENTER_SHORT and self._pos_dir == 0:
            # Enter short
            lots = [0.01, 0.04, 0.08][size_level]
            sl_dist = SL_MULTS[sl_mult_idx] * max(atr, 0.01)
            tp_dist_mult = TP_MULTS[tp_mult_idx]
            self._pos_dir = -1
            self._pos_entry = bid
            self._pos_sl = bid + sl_dist
            self._pos_tp = bid - tp_dist_mult * sl_dist
            self._pos_lots = lots
            self._pos_bars = 0
            self._pos_risk_per_unit = sl_dist

        # --- Check SL/TP/time-stop on existing position ---
        if self._pos_dir != 0:
            self._pos_bars += 1

            # Check SL hit
            if self._pos_dir == 1 and bid <= self._pos_sl:
                close_pnl = (self._pos_sl - self._pos_entry) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(False)
                self._trades.append({"entry": self._pos_entry, "exit": self._pos_sl, "dir": 1,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "SL"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

            elif self._pos_dir == -1 and ask >= self._pos_sl:
                close_pnl = (self._pos_entry - self._pos_sl) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(False)
                self._trades.append({"entry": self._pos_entry, "exit": self._pos_sl, "dir": -1,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "SL"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

            # Check TP hit
            elif self._pos_dir == 1 and bid >= self._pos_tp:
                close_pnl = (self._pos_tp - self._pos_entry) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(True)
                self._trades.append({"entry": self._pos_entry, "exit": self._pos_tp, "dir": 1,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "TP"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

            elif self._pos_dir == -1 and ask <= self._pos_tp:
                close_pnl = (self._pos_entry - self._pos_tp) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(True)
                self._trades.append({"entry": self._pos_entry, "exit": self._pos_tp, "dir": -1,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "TP"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

            # Time-stop
            elif self._pos_bars >= self.time_stop_bars:
                if self._pos_dir == 1:
                    close_pnl = (bid - self._pos_entry) * self._pos_lots * self.spec.contract_size
                else:
                    close_pnl = (self._pos_entry - ask) * self._pos_lots * self.spec.contract_size
                close_pnl_acct = self.acct.to_account_ccy(close_pnl, self.spec.currency_profit)
                self._equity += close_pnl_acct
                self._peak_equity = max(self._peak_equity, self._equity)
                self._recent_wins.append(close_pnl_acct > 0)
                self._trades.append({"entry": self._pos_entry, "exit": price, "dir": self._pos_dir,
                                     "lots": self._pos_lots, "pnl": close_pnl_acct, "bars": self._pos_bars,
                                     "reason": "TIME_STOP"})
                self._pos_dir = 0; self._pos_bars = 0; trade_closed = True

        # --- Compute reward ---
        # Balanced reward: encourage trading while controlling risk
        # Key insight: if penalties are too harsh, agent learns to never trade (hold = 0 reward)
        equity_frac = self._equity / max(self._starting_equity, 1.0)
        drawdown = (self._peak_equity - self._equity) / max(self._peak_equity, 1.0)

        reward = 0.0

        if len(self._trades) > 0 and trade_closed:
            last_pnl = self._trades[-1]["pnl"]
            # Risk-adjusted PnL: normalize by equity
            pnl_frac = last_pnl / max(self._equity, 1.0)
            reward = pnl_frac * 5.0  # moderate scale

            # Symmetric win/loss signal — let agent learn from outcomes
            if last_pnl > 0:
                reward += 0.3  # win bonus
            else:
                reward -= 0.3  # loss penalty (symmetric)

            # Reward for good R-multiple (TP hits are better than time-stops)
            reason = self._trades[-1].get("reason", "")
            if reason == "TP":
                reward += 0.2  # bonus for hitting TP
            elif reason == "TIME_STOP":
                reward -= 0.1  # small penalty for time-stop (indecisive)
        else:
            # Small holding reward for being in profit (R-multiple based)
            if self._pos_dir != 0 and self._pos_risk_per_unit > 0:
                r_dist = (price - self._pos_entry) if self._pos_dir == 1 else (self._pos_entry - price)
                cur_r = r_dist / self._pos_risk_per_unit
                reward = cur_r * 0.01  # per-step signal for being in profit

        # Drawdown penalty — LINEAR, kicks in at 5%, capped
        # Old exponential was too harsh: 85% DD → 361 penalty per step
        # New: gentle linear penalty that increases with DD
        if drawdown > 0.05:
            dd_penalty = min((drawdown - 0.05) * 5.0, 3.0)  # 5%→0, 10%→0.25, 20%→0.75, 50%→2.25, capped at 3.0
            reward -= dd_penalty

        # Equity wipeout — hard penalty
        if self._equity <= self._starting_equity * 0.5:
            reward -= 5.0

        # Track equity history
        self._equity_history.append(self._equity)

        # Advance to next bar
        self._bar_idx += 1
        terminated = self._bar_idx >= self.n_bars
        truncated = self._equity <= self._starting_equity * 0.5  # 50% equity = stop

        # Update signal engine state (for feature computation on next step)
        if not terminated:
            # Skip _evaluate_row — observation is built directly from numpy arrays
            pass

        obs = self._get_obs()
        info = self._get_info()
        info["trade_closed"] = trade_closed
        info["close_pnl"] = close_pnl if trade_closed else 0.0

        if self.render_mode == "human":
            self._render_human(info)

        return obs, reward, terminated, truncated, info

    def _render_human(self, info: dict):
        """Print current state to terminal."""
        if info.get("trade_closed"):
            pnl = info.get("close_pnl", 0.0)
            print(f"  bar={self._bar_idx} eq={self._equity:.0f} "
                  f"dd={info['drawdown']:.1%} trades={info['n_trades']} "
                  f"wr={info['win_rate']:.0%} pnl={pnl:+.2f}")

    def get_metrics(self) -> dict:
        """Compute backtest metrics from trade history."""
        if not self._trades:
            return {"n_trades": 0}

        pnls = [t["pnl"] for t in self._trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

        # Sharpe ratio (annualized, assuming 252 trading days × 24 hours)
        if len(pnls) > 1:
            returns = np.array(pnls) / self._starting_equity
            sharpe = np.mean(returns) / max(np.std(returns), 1e-9) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0

        # Max drawdown
        equity_curve = np.cumsum([self._starting_equity] + pnls)
        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / peak
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        return {
            "n_trades": len(pnls),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "final_equity": self._equity,
            "return_pct": (self._equity - self._starting_equity) / self._starting_equity,
        }
