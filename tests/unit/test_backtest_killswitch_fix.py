"""Regression tests: research backtests must not be truncated by live guardrails.

The backtest broker used to inherit GuardrailConfig's default drawdown
kill-switches (3% daily / 8% total). A losing stretch would silently freeze
the backtest, hiding the strategy's true performance. Research backtests must
see the FULL trade history; only the live paper/demo loops keep the kill-switch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from slytrade.backtest.engine import BacktestConfig
from slytrade.backtest.reporting import run_managed_aligned_backtest_from_bars
from slytrade.backtest.trade_management import TradeManagementConfig
from slytrade.strategies.personality_adaptive import PersonalityAdaptiveConfig
from slytrade.tasks import _persona_config_from_risk, _trade_config_from_risk


def make_oscillating_bars(n: int = 400) -> pd.DataFrame:
    """Random-walk market whose footprint alternates bull/bear in blocks.

    The persona alternates long/short on these, wins ~half the time, and loses
    to spread + commission — enough closed trades to prove no kill-switch
    truncation without needing a directional edge.
    """
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    bull = np.tile(np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), n // 10 + 1)[:n]
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "decision_time": pd.date_range("2026-01-01 00:01", periods=n, freq="min", tz="UTC"),
            "quote_bid": close - 0.1,
            "quote_ask": close + 0.1,
            "quote_time": pd.date_range("2026-01-01 00:01", periods=n, freq="min", tz="UTC"),
            "quote_is_fresh": True,
            "atr": 0.2,
            "atr_norm": 0.002,
            "bos_dir": bull,
            "choch_dir": 0.0,
            "liquidity_sweep": -bull,
            "fvg_bullish": (bull > 0).astype(float),
            "fvg_bearish": (bull < 0).astype(float),
            "order_block_bullish": (bull > 0).astype(float),
            "order_block_bearish": (bull < 0).astype(float),
            "premium_discount": -0.5 * bull,
            "trend_strength": 0.5 * bull,
            "tick_rate_per_second": 2.0,
            "quote_spread": 0.2,
            "session_asia": 0.0,
            "session_london": 1.0,
            "session_ny_am": 0.0,
            "session_ny_pm": 0.0,
            "session_other": 0.0,
            "mtf_bias": bull,
            "mtf_confluence_score": 3.0,
        }
    )


def test_default_backtest_config_disables_drawdown_killswitch() -> None:
    cfg = BacktestConfig()
    assert cfg.max_daily_drawdown is None
    assert cfg.max_total_drawdown is None


def test_losing_backtest_is_not_truncated_by_killswitch() -> None:
    # Oscillating signals -> the persona alternates entries and loses to
    # spread + commission. The old broker's default drawdown guard would have
    # frozen it after a few losses; the research backtest must record the full
    # history (every round trip), losses included.
    bars = make_oscillating_bars(400)
    result = run_managed_aligned_backtest_from_bars(
        bars,
        strategy_name="persona-adaptive",
        symbol="XAUUSD",
        point_value=1.0,
        config=BacktestConfig(point_value=1.0, initial_balance=100_000.0),
        trade_config=TradeManagementConfig(stop_loss_atr=1.0, take_profit_atr=2.0),
        persona_config=PersonalityAdaptiveConfig(
            point_value=1.0,
            risk_based_sizing=False,
            require_sweep_reversal=False,
            require_entry_momentum=False,
            strict_mtf_direction=False,
            use_regime_filter=False,
            cooldown_bars=1,
        ),
    )
    # Even a strategy that loses money must have been allowed to keep trading:
    # with a kill-switch active this would be frozen at a handful of fills.
    assert len(result.trades) >= 10
    # And it should have realized losses (no artificial guardrail "profit").
    realized = [r.realized_pnl for r in result.trades]
    assert sum(realized) < 0


def test_risk_config_loaders_handle_null_trailing() -> None:
    trade = _trade_config_from_risk()
    assert trade.stop_loss_atr > 0
    assert trade.take_profit_atr > 0
    # trailing_atr_mult: null in risk.yaml -> no trailing stop.
    assert trade.trailing_stop_atr is None
    persona = _persona_config_from_risk()
    assert persona.strict_mtf_direction is True
    assert persona.require_entry_momentum is True
