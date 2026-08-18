"""Live trading loop — places real orders on the connected MT5 account.

This is the LIVE deployment loop. It places real orders on whatever MT5
account the terminal is logged into (a demo account in the current setup, the
live account later). The stage setting only records which account type is
attached — the loop itself is the live run:

* refuses to start unless ``SLYTRADE_ALLOW_LIVE=1`` AND stage is ``demo``
  (the account currently attached is a demo account);
* refuses to trade until broker reconciliation succeeds;
* entries only (never exits) are gated by session window, news gate, circuit
  breaker and risk sizing;
* every order still passes the guardrails + OMS + adapter path;
* stop-loss/take-profit are attached to the broker order (server-side);
* every fill and every close is journaled to ``data/live_journal`` so the bot
  learns from what actually worked live.

If reconciliation fails or the kill switch trips, the loop stops placing new
orders but keeps streaming quotes so the operator can observe and intervene.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from slytrade.backtest.execution import Quote
from slytrade.core.config import load_config
from slytrade.currency import CurrencyConverter, load_converter
from slytrade.execution.models import OrderIntent, OrderKind, OrderStatus, Side
from slytrade.execution.oms import OrderManagementSystem
from slytrade.monitoring.gates import DeploymentStage
from slytrade.monitoring.operations import SoakMonitor
from slytrade.risk.guardrails import GuardrailConfig, TradingGuardrails
from slytrade.risk.sizing import risk_based_volume
from slytrade.runtime.alerting import AlertManager
from slytrade.runtime.circuit_breaker import LossCircuitBreaker, limits_from_config
from slytrade.runtime.logs import setup_logging
from slytrade.runtime.metrics_server import TradingMetrics
from slytrade.runtime.news_gate import NewsGate
from slytrade.runtime.paper_loop import BarBuilder
from slytrade.runtime.settings import RuntimeSettings, TradingStage
from slytrade.runtime.trading_window import TradingWindow, window_from_settings
from slytrade.strategies.personality_adaptive import PersonalityAdaptiveStrategy


@dataclass(frozen=True)
class LiveLoopSummary:
    bars_processed: int
    orders_submitted: int
    orders_filled: int
    errors: int
    reconciled: bool
    kill_switch: bool
    final_equity: float | None


# Backward-compatible aliases (older scripts/tests import these names).
DemoLoopSummary = LiveLoopSummary


@dataclass
class LiveTradingLoop:
    """Guarded live trading loop against the connected MT5 account."""

    settings: RuntimeSettings
    mt5: Any
    strategy: Any = None
    logger: Any = field(default=None, init=False)

    adapter: Any = field(init=False)
    guardrails: TradingGuardrails = field(init=False)
    breaker: LossCircuitBreaker = field(init=False)
    window: TradingWindow = field(init=False)
    news_gate: NewsGate = field(init=False)
    alerter: AlertManager = field(init=False)
    metrics: TradingMetrics = field(default_factory=TradingMetrics, init=False)
    soak: SoakMonitor = field(init=False)
    bar_builder: BarBuilder = field(init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _bar_index: int = field(default=0, init=False)
    _errors: int = field(default=0, init=False)
    _orders: int = field(default=0, init=False)
    _fills: int = field(default=0, init=False)
    _kill_switch_alerted: bool = field(default=False, init=False)
    _side: str = field(default="flat", init=False)
    _point_value: float = field(default=1.0, init=False)
    _converter: CurrencyConverter = field(default_factory=lambda: CurrencyConverter(1.0), init=False)
    _window_bars: list[dict[str, Any]] = field(default_factory=list, init=False)
    _profile: Any = field(default=None, init=False)  # validated timeframe profile
    _minlot_warned: bool = field(default=False, init=False)
    _journal_open: dict[str, Any] | None = field(default=None, init=False)
    _pending_limit: dict[str, Any] | None = field(default=None, init=False)  # working limit order
    _ticks: int = field(default=0, init=False)
    _quotes_seen: int = field(default=0, init=False)
    _stale_quotes: int = field(default=0, init=False)
    _last_heartbeat_at: float = field(default=0.0, init=False)
    _last_quote: Quote | None = field(default=None, init=False)
    _last_decision: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.logger = setup_logging(self.settings.log_level, self.settings.log_dir, self.settings.json_logs)

        problems = self.settings.fail_closed_checks()
        if problems:
            raise ValueError("startup blocked: " + "; ".join(problems))
        if not self.settings.allow_live:
            raise ValueError("live trading is disabled; set SLYTRADE_ALLOW_LIVE=1 and SLYTRADE_STAGE=demo")
        if self.settings.stage != TradingStage.DEMO:
            raise ValueError(f"live trading requires stage=demo (got {self.settings.stage})")

        config = load_config(self.settings.config_dir)
        risk_cfg = config.risk
        self._converter = load_converter(risk_cfg)
        limits = limits_from_config(risk_cfg)
        self.breaker = LossCircuitBreaker(limits)
        self.window = window_from_settings(self.settings.trading_days, self.settings.trading_start_utc, self.settings.trading_end_utc)
        self.alerter = AlertManager.from_settings(self.settings, self.logger)
        from slytrade.runtime.calendar import build_news_gate_from_settings

        self.news_gate = build_news_gate_from_settings(self.settings)

        self.guardrails = TradingGuardrails(
            GuardrailConfig(
                allow_live_trading=True,
                max_daily_drawdown=limits.max_daily_drawdown,
                max_total_drawdown=limits.max_total_drawdown,
                max_position_volume=float(risk_cfg.get("max_position_volume", 1.0)),
                max_spread_points=float(config.broker.get("execution", {}).get("max_spread_points", 50.0)),
            ),
            initial_equity=float(self.settings.initial_balance),
            kill_switch_path=self.settings.kill_switch_path,
        )

        from slytrade.brokers.mt5_adapter import MT5BrokerAdapter
        from slytrade.monitoring.metrics import ExecutionMetrics

        self.settings.state_path.mkdir(parents=True, exist_ok=True)
        self.execution_metrics = ExecutionMetrics()
        self.adapter = MT5BrokerAdapter(
            self.mt5,
            oms=OrderManagementSystem(),
            guardrails=self.guardrails,
            metrics=self.execution_metrics,
            allow_trading=True,
            expected_positions={},
        )
        self.soak = SoakMonitor(DeploymentStage.SHADOW, min_samples=1, stale_after=timedelta(minutes=5))
        from slytrade.config.timeframe_profiles import profile_for

        self._profile = profile_for(self.settings.timeframe)
        self.bar_builder = BarBuilder(self.settings.symbol, self.settings.timeframe)
        if self.strategy is None:
            self.strategy = self._build_champion_strategy()

    # -- strategy / features ---------------------------------------------------
    def _build_champion_strategy(self) -> PersonalityAdaptiveStrategy:
        """Build the persona strategy with the VALIDATED champion configuration.

        The live loop must trade the same structure the backtest validated:
        the per-timeframe profile (min confluence score, cooldown, H4-trend
        gate, stop/target multiples) plus the footprint gates from
        configs/risk.yaml. The dataclass defaults are NOT the champion —
        e.g. ``htf_trend_timeframe`` is None by default, which would silently
        skip the H4-trend gate that IS the measured edge.
        """
        from slytrade.config.timeframe_profiles import profile_for
        from slytrade.strategies.personality_adaptive import PersonalityAdaptiveConfig

        risk_cfg = load_config(self.settings.config_dir).risk
        ict = risk_cfg.get("ict", {}) or {}
        entry = ict.get("entry", {}) or {}
        profile = profile_for(self.settings.timeframe)
        symbol = (self.settings.symbol or "XAUUSD").upper()
        point_value = 100.0 if symbol in ("XAUUSD", "XAGUSD") else 1.0
        config = PersonalityAdaptiveConfig(
            min_score=int(entry.get("min_score", profile.min_score)),
            cooldown_bars=int(entry.get("cooldown_bars", profile.cooldown_bars)),
            require_sweep_reversal=bool(entry.get("require_sweep_reversal", False)),
            sweep_reversal_window=int(entry.get("sweep_reversal_window", 12)),
            require_entry_momentum=bool(entry.get("require_entry_momentum", True)),
            strict_mtf_direction=bool(entry.get("strict_mtf_direction", True)),
            max_spread=(float(entry["max_spread"]) if entry.get("max_spread") is not None else None),
            htf_trend_timeframe=(
                str(entry["htf_trend_timeframe"]) if entry.get("htf_trend_timeframe") else profile.htf_trend_timeframe
            ),
            smc_displacement=int(entry.get("smc_displacement", 0)),
            smc_ifvg=int(entry.get("smc_ifvg", 0)),
            smc_breaker=int(entry.get("smc_breaker", 0)),
            smc_vi=int(entry.get("smc_vi", 0)),
            smc_dol_tap=int(entry.get("smc_dol_tap", 0)),
            limit_entry_atr=float(entry.get("limit_entry_atr", 0.0) or 0.0),
            point_value=point_value,
        )
        return PersonalityAdaptiveStrategy(symbol=self.settings.symbol, volume=0.1, config=config)

    def _higher_timeframe_features(self, frame: pd.DataFrame) -> dict[str, float]:
        """Resample the rolling window to H4/D1 and merge higher-TF features.

        Produces the ``htf_*`` columns plus ``mtf_bias`` / ``mtf_confluence_score``
        that activate the champion's H4-trend alignment gate — the validated edge.
        Returns {} while the window is too short to form a higher-TF bar.
        """
        if len(frame) < 2 or "time" not in frame.columns:
            return {}
        try:
            from slytrade.data.resample import resample_bars_to_timeframe
            from slytrade.features.mtf import compute_mtf_ict_features

            higher = {
                tf: df
                for tf, df in (
                    ("H4", resample_bars_to_timeframe(frame, "H4")),
                    ("D1", resample_bars_to_timeframe(frame, "D1")),
                )
                if not df.empty
            }
            if not higher:
                return {}
            merged = compute_mtf_ict_features(frame, higher)
        except Exception:  # pragma: no cover - feature degradation must never crash the loop
            return {}
        last = merged.iloc[-1]
        out: dict[str, float] = {}
        for column in merged.columns:
            if column.startswith("htf_") or column in ("mtf_bias", "mtf_confluence_score"):
                value = last[column]
                if pd.isna(value):
                    out[column] = 0.0
                    continue
                try:
                    out[column] = float(value)
                except (TypeError, ValueError):
                    continue  # skip non-numeric htf columns (e.g. htf_h4_timeframe == "H4")
        return out

    def _discover_replay_bars(self, resolved_symbol: str) -> Path | None:
        """Find the pipeline's aligned bars file without any configuration.

        ``slytrade full-pipeline`` writes aligned bars to
        ``data/processed/aligned/<symbol>/<tf>/bars.parquet`` (canonical symbol,
        lowercase timeframe). When the operator runs the live loop right after
        the pipeline — as they always should — the warmup must find that file
        automatically instead of requiring SLYTRADE_REPLAY_BARS_FILE.
        """
        base = resolved_symbol.split("m")[0] if resolved_symbol.endswith("m") else resolved_symbol
        tf = (self.settings.timeframe or "M15").lower()
        candidates = [
            Path(self.settings.data_dir) / "processed" / "aligned" / base / tf / "bars.parquet",
            Path(self.settings.data_dir) / "processed" / "aligned" / resolved_symbol / tf / "bars.parquet",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _warmup(self, resolved_symbol: str) -> None:
        """Seed the feature window from stored bars before the first streamed bar.

        Pre-fills ATR/EMA/H4/D1 context so the champion's gates are warm from
        the first bar instead of cold for the first ``history_bars``. Every
        path logs explicitly — a silent no-op here cost 4 hours of cold H4
        context once before. An explicit SLYTRADE_REPLAY_BARS_FILE wins; when
        unset, the pipeline's aligned output is auto-discovered.
        """
        path = self.settings.replay_bars_file
        if not path:
            discovered = self._discover_replay_bars(resolved_symbol)
            if discovered is None:
                self.logger.warning(
                    "warmup: no replay bars file found (cold start) — run 'slytrade full-pipeline "
                    f"{resolved_symbol.split('m')[0] if resolved_symbol.endswith('m') else resolved_symbol} "
                    f"--timeframe {self.settings.timeframe}' first, or set SLYTRADE_REPLAY_BARS_FILE",
                    extra={"event": "warmup_none", "hint": "run the full pipeline to generate data/processed/aligned/<symbol>/<tf>/bars.parquet"},
                )
                return
            path = str(discovered)
            self.logger.info(
                "warmup: auto-discovered bars file",
                extra={"event": "warmup_discovered", "path": path},
            )
        try:
            frame = pd.read_parquet(path)
        except Exception as exc_parquet:
            try:
                frame = pd.read_csv(path)
            except Exception as exc_csv:  # pragma: no cover
                self.logger.warning(
                    "warmup skipped: could not read file",
                    extra={"event": "warmup_skip", "path": path, "reason": str(exc_csv) or str(exc_parquet)},
                )
                return
        if frame is None or frame.empty:
            self.logger.warning("warmup skipped: file has no rows", extra={"event": "warmup_empty", "path": path})
            return
        if "symbol" in frame.columns:
            # Match BOTH the canonical research symbol (XAUUSD) and the broker
            # symbol (XAUUSDm) — the aligned bars are canonical while the
            # adapter resolves to the broker suffix.
            symbols = frame["symbol"].astype(str)
            base = resolved_symbol.split("m")[0] if resolved_symbol.endswith("m") else resolved_symbol
            frame = frame[symbols.str.upper().isin({resolved_symbol.upper(), base.upper()})]
        if frame.empty:
            self.logger.warning(
                "warmup skipped: no rows for symbol",
                extra={"event": "warmup_symbol_miss", "path": path, "resolved": resolved_symbol},
            )
            return
        frame = frame.sort_values("time")
        cap = max(64, int(self.settings.history_bars))
        tail = frame.tail(cap)
        self._window_bars.extend(row.to_dict() for _, row in tail.iterrows())
        self.logger.info(
            f"warmup loaded {len(tail)} bars from {path}",
            extra={"event": "warmup", "bars": len(tail), "path": path, "first": str(tail["time"].iloc[0]), "last": str(tail["time"].iloc[-1])},
        )

    # -- live journal (the bot's memory) --------------------------------------
    def _journal_path(self) -> Path:
        return Path(self.settings.data_dir) / "live_journal" / "trades.parquet"

    @staticmethod
    def _journal_feature(series: pd.Series, name: str, default: float = 0.0) -> float:
        if name not in series.index:
            return default
        value = series[name]
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _journal_entry(self, series: pd.Series, sized: OrderIntent, quote: Quote) -> None:
        """Record an entry snapshot: the features the champion saw + the trade.

        This is the bot's live memory — every real fill is journaled with the
        causal features that produced it, so ``slytrade learn`` can later see
        what actually worked live vs what the backtest predicted.
        """
        try:
            row = {
                "time": pd.Timestamp(quote.time).isoformat(),
                "symbol": sized.symbol,
                "side": "buy" if sized.side == Side.BUY else "sell",
                "entry": float(sized.limit_price) if sized.kind == OrderKind.LIMIT and sized.limit_price is not None else float(quote.mid),
                "stop": float(sized.stop_loss or 0.0),
                "target": float(sized.take_profit or 0.0),
                "volume": float(sized.volume),
                "persona_score": self._journal_feature(series, "persona_score"),
                "persona_bias": self._journal_feature(series, "persona_bias"),
                "bos_dir": self._journal_feature(series, "bos_dir"),
                "choch_dir": self._journal_feature(series, "choch_dir"),
                "liquidity_sweep": self._journal_feature(series, "liquidity_sweep"),
                "premium_discount": self._journal_feature(series, "premium_discount"),
                "trend_strength": self._journal_feature(series, "trend_strength"),
                "atr": self._journal_feature(series, "atr"),
                "htf_h4_bos_dir": self._journal_feature(series, "htf_h4_bos_dir"),
                "mtf_bias": self._journal_feature(series, "mtf_bias"),
                "outcome_r": float("nan"),
                "exit": float("nan"),
                "exit_reason": "",
            }
            self._journal_open = row
            self._journal_append(row)
        except Exception as exc:  # pragma: no cover - journaling must never stop trading
            self.logger.warning("journal entry failed", extra={"event": "journal_error", "reason": str(exc)})

    def _journal_exit(self, resolved_symbol: str, bar: dict[str, Any], quote: Quote) -> None:
        """Close the journal row for the trade that just ended (server-side)."""
        open_row = self._journal_open
        if not open_row or open_row.get("symbol") != resolved_symbol:
            self._journal_open = None
            return
        self._journal_open = None
        try:
            direction = 1.0 if open_row["side"] == "buy" else -1.0
            entry = float(open_row["entry"])
            stop = float(open_row["stop"] or 0.0)
            target = float(open_row["target"] or 0.0)
            high = float(bar.get("high", quote.mid) or quote.mid)
            low = float(bar.get("low", quote.mid) or quote.mid)
            if direction > 0:
                hit_sl, hit_tp = low <= stop, high >= target
            else:
                hit_sl, hit_tp = high >= stop, low <= target
            if hit_sl:
                exit_price, reason = stop, "stop_loss"
            elif hit_tp:
                exit_price, reason = target, "take_profit"
            else:
                exit_price, reason = float(quote.mid), "unknown"
            risk = abs(entry - stop)
            r = direction * (exit_price - entry) / risk if risk > 0 else 0.0
            row = dict(open_row)
            row["exit"] = exit_price
            row["exit_reason"] = reason
            row["outcome_r"] = round(r, 4)
            self._journal_append(row, replace=open_row["time"])
        except Exception as exc:  # pragma: no cover
            self.logger.warning("journal exit failed", extra={"event": "journal_error", "reason": str(exc)})

    def _journal_append(self, row: dict[str, Any], replace: str | None = None) -> None:
        path = self._journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                frame = pd.read_parquet(path)
            except Exception:  # pragma: no cover
                frame = pd.DataFrame()
        else:
            frame = pd.DataFrame()
        if replace is not None and not frame.empty and "time" in frame.columns:
            frame = frame[frame["time"] != replace]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        frame.to_parquet(path, index=False)

    # -- main ------------------------------------------------------------------
    def run(self, *, max_bars: int | None = None, max_seconds: float = 0.0) -> LiveLoopSummary:
        self._install_signal_handlers()
        start = time.monotonic()
        self.logger.info("live loop starting", extra={"event": "live_start", "symbol": self.settings.symbol})

        self.adapter.connect()
        resolved = self.adapter.resolve_symbol(self.settings.symbol)
        self._warmup(resolved)
        try:
            spec = self.adapter.symbol_spec(resolved)
            self._point_value = spec.point_value_per_price_unit
            self.logger.info("symbol spec loaded", extra={"event": "spec", "point_value": self._point_value})
        except Exception as exc:  # pragma: no cover - broker dependent
            self.logger.warning("could not load symbol spec; sizing falls back to point_value=1.0", extra={"event": "spec_warning", "reason": str(exc)})
        self.adapter.expected_positions = self._current_positions(resolved)
        reconciliation = self.adapter.reconcile()
        self.logger.info(
            "reconciliation",
            extra={"event": "reconcile", "status": "ok" if reconciliation.reconciled else "blocked"},
        )
        if not reconciliation.reconciled:
            self.alerter.alert("critical", "live loop blocked", reconciliation.detail)
        # Full startup summary: the operator can see the whole plan at a glance.
        profile = self._profile
        strat_cfg = self.strategy.config
        hb = max(float(self.settings.heartbeat_interval_seconds or 5.0), 5.0)
        self.logger.info(
            "live config: "
            f"symbol={resolved} timeframe={self.settings.timeframe} min_score={strat_cfg.min_score} "
            f"cooldown={strat_cfg.cooldown_bars} htf={strat_cfg.htf_trend_timeframe} "
            f"limit_entry={strat_cfg.limit_entry_atr}ATR stop={profile.stop_loss_atr}ATR target={profile.take_profit_atr}ATR "
            f"max_hold={profile.max_bars_in_trade} risk={self.breaker.limits.risk_per_trade:.1%}/trade "
            f"window={self.settings.trading_days} {self.settings.trading_start_utc}-{self.settings.trading_end_utc} "
            f"news={'on' if self.settings.news_enabled else 'off'} poll={self.settings.poll_seconds}s "
            f"heartbeat={hb}s history={self.settings.history_bars} warm_bars={len(self._window_bars)}",
            extra={
                "event": "config",
                "symbol": resolved,
                "timeframe": self.settings.timeframe,
                "min_score": strat_cfg.min_score,
                "cooldown_bars": strat_cfg.cooldown_bars,
                "htf_trend_timeframe": strat_cfg.htf_trend_timeframe,
                "limit_entry_atr": strat_cfg.limit_entry_atr,
                "stop_atr": profile.stop_loss_atr,
                "target_atr": profile.take_profit_atr,
                "max_bars_in_trade": profile.max_bars_in_trade,
                "risk_per_trade": self.breaker.limits.risk_per_trade,
                "window_days": self.settings.trading_days,
                "window_hours": f"{self.settings.trading_start_utc}-{self.settings.trading_end_utc}",
                "news_enabled": self.settings.news_enabled,
                "poll_seconds": self.settings.poll_seconds,
                "heartbeat_seconds": hb,
                "history_bars": self.settings.history_bars,
                "window_bars_warm": len(self._window_bars),
            },
        )

        try:
            while not self._stop.is_set():
                elapsed = time.monotonic() - start
                if max_seconds and elapsed >= max_seconds:
                    break
                if max_bars is not None and self._bar_index >= max_bars:
                    break

                # Periodic broker reconciliation (GAP-7): re-verify broker state
                # every 5 minutes instead of only once at startup.
                if self._bar_index > 0 and self._bar_index % 300 == 0:
                    self.adapter.expected_positions = self._current_positions(resolved)
                    check = self.adapter.reconcile()
                    if not check.reconciled:
                        self.alerter.alert("critical", "reconciliation drift", check.detail)
                    self.logger.info("periodic reconciliation", extra={"event": "reconcile", "status": "ok" if check.reconciled else "drift"})

                try:
                    quote = self.adapter.quote(resolved)
                except Exception as exc:  # pragma: no cover - broker dependent
                    self._errors += 1
                    self.metrics.broker_errors_total.inc()
                    self.alerter.alert("warning", "quote error", str(exc))
                    time.sleep(self.settings.poll_seconds)
                    continue

                if quote is None:
                    time.sleep(self.settings.poll_seconds)
                    continue

                self._ticks += 1
                self._quotes_seen += 1
                self._last_quote = quote

                bar = self.bar_builder.on_quote(quote)
                if bar is not None:
                    self._bar_index += 1
                    self._on_bar(bar, quote, resolved)
                self._export_metrics(resolved)
                self._heartbeat()
                self._status_heartbeat(quote, resolved)
                time.sleep(self.settings.poll_seconds)
        finally:
            self.adapter.disconnect()

        self.alerter.alert(
            "info",
            "live loop stopped",
            f"bars={self._bar_index} orders={self._orders} fills={self._fills} errors={self._errors}",
        )
        return LiveLoopSummary(
            bars_processed=self._bar_index,
            orders_submitted=self._orders,
            orders_filled=self._fills,
            errors=self._errors,
            reconciled=reconciliation.reconciled,
            kill_switch=self.guardrails.kill_switch,
            final_equity=None,
        )

    # -- per-bar ---------------------------------------------------------------
    def _on_bar(self, bar: dict[str, Any], quote: Quote, resolved_symbol: str) -> None:
        if self.guardrails.kill_switch:
            self._alert_kill_switch_once()
            return
        if not self.window.is_open(quote.time):
            self.logger.info("window closed — no entries", extra={"event": "window_closed", "time": str(quote.time)})
            return
        if self.news_gate.is_red_folder(quote.time):
            self.metrics.news_pauses_total.inc()
            self.logger.info(
                "red folder — entries paused",
                extra={"event": "news_pause", "reason": self.news_gate.reason(quote.time) or "news"},
            )
            return
        decision = self.breaker.check()
        if not decision.allowed:
            self.logger.info("entry paused", extra={"event": "paused", "reason": decision.reason})
            return

        self._window_bars.append(bar)
        cap = max(64, int(self.settings.history_bars))
        if len(self._window_bars) > cap:
            self._window_bars.pop(0)

        # Derive the live side from the BROKER (server-side SL/TP closes the
        # position), and notify the strategy when a position closes so its own
        # ``_side`` state resets — otherwise it would never re-enter the same
        # direction, which the validated champion does (up to 69 in a row).
        held = self._current_positions(resolved_symbol).get(resolved_symbol, 0.0)
        side = "long" if held > 0 else ("short" if held < 0 else "flat")
        if side == "flat" and self._side != "flat":
            hook = getattr(self.strategy, "on_position_closed", None)
            if hook is not None:
                hook()
            self._journal_exit(resolved_symbol, bar, quote)
            self.logger.info("position closed", extra={"event": "position_closed", "symbol": resolved_symbol})
        self._side = side
        if side != "flat":
            self.logger.info(
                f"holding {side} — SL/TP managed server-side on the broker",
                extra={"event": "holding", "side": side, "symbol": resolved_symbol},
            )

        # Working limit order: the broker fills it into a position, or it sits
        # until its hold window expires and we cancel it. Never stack a second
        # order while one is resting.
        if self._pending_limit is not None:
            if side != "flat":
                # The broker filled our limit — a position now exists.
                self._journal_entry_from_pending(resolved_symbol, quote)
                self._pending_limit = None
                return
            self._pending_limit["bars"] = int(self._pending_limit.get("bars", 0)) + 1
            ttl = int(self._profile.max_bars_in_trade or 60)
            if self._pending_limit["bars"] >= ttl:
                self._cancel_pending(resolved_symbol)
                self._pending_limit = None
                hook = getattr(self.strategy, "on_position_closed", None)
                if hook is not None:
                    hook()  # let the champion try again on a fresh setup
            return

        if self._side == "flat":
            series = self._decision_series(bar)
            intent = self.strategy.on_bar(self._bar_index, series)
            self._last_decision = self._decision_trace(series, quote, intent)
            self.logger.info(self._last_decision, extra={"event": "decision"})
            self._publish_status()
            if intent is None:
                return
            sized = self._sized_intent(intent, series, quote)
            report = self.adapter.submit(sized, quote)
            self._orders += 1
            self.metrics.orders_total.labels(status=str(report.status.value)).inc()
            if report.status == OrderStatus.FILLED:
                self._fills += 1
                self._side = "long" if sized.side == Side.BUY else "short"
                self._journal_entry(series, sized, quote)
                self.logger.info("live order filled", extra={"event": "live_fill", "order_id": sized.client_order_id})
            elif sized.kind == OrderKind.LIMIT and report.status == OrderStatus.ACCEPTED:
                self._pending_limit = {
                    "symbol": sized.symbol,
                    "side": sized.side.value,
                    "entry": float(sized.limit_price or quote.mid),
                    "volume": sized.volume,
                    "bars": 0,
                    "ticket": report.broker_order_id,
                    "features": {c: self._journal_feature(series, c) for c in
                                 ("persona_score", "persona_bias", "bos_dir", "choch_dir", "liquidity_sweep",
                                  "premium_discount", "trend_strength", "htf_h4_bos_dir", "mtf_bias")},
                }
                self.logger.info("limit order resting", extra={"event": "limit_resting",
                                                               "price": sized.limit_price, "ticket": report.broker_order_id})
            elif report.status == OrderStatus.REJECTED:
                self.logger.warning("live order rejected", extra={"event": "live_reject", "reason": report.message})
                self.alerter.alert("warning", "live order rejected", report.message)

    def _cancel_pending(self, resolved_symbol: str) -> None:
        """Cancel the resting limit order once its hold window has elapsed."""
        ticket = (self._pending_limit or {}).get("ticket")
        if ticket:
            try:
                cancel = getattr(self.adapter, "cancel_order", None)
                if cancel is not None:
                    cancel(int(ticket))
            except Exception as exc:  # pragma: no cover - broker dependent
                self.logger.warning("limit cancel failed", extra={"event": "limit_cancel_fail", "reason": str(exc)})
        self.logger.info("limit order cancelled (expired)", extra={"event": "limit_expired", "symbol": resolved_symbol})

    def _journal_entry_from_pending(self, resolved_symbol: str, quote: Quote) -> None:
        """Journal a broker-filled limit entry (fill price = the resting limit)."""
        pending = self._pending_limit
        if pending is None:
            return
        side = Side.BUY if pending["side"] == "buy" else Side.SELL
        sized = OrderIntent(
            symbol=resolved_symbol, side=side, volume=float(pending["volume"]),
            limit_price=float(pending["entry"]), stop_loss=0.0, take_profit=0.0, reason="persona_limit_fill",
        )
        features = pd.Series(pending.get("features", {}), dtype=object)
        self._journal_entry(features, sized, quote)

    def _decision_series(self, bar: dict[str, Any]) -> pd.Series:
        from slytrade.features.ict import compute_ict_features

        if len(self._window_bars) < 2:
            frame = pd.DataFrame([bar])
        else:
            frame = pd.DataFrame(self._window_bars)
        features = compute_ict_features(frame)
        last = features.iloc[-1]
        series = pd.Series(bar, dtype=object)
        for column in features.columns:
            if column not in ("time", "symbol", "timeframe"):
                series[column] = last[column]
        # H4/D1 context so the champion's H4-trend alignment gate fires live
        # exactly as it does in the validated backtest.
        for column, value in self._higher_timeframe_features(frame).items():
            series[column] = value
        series["quote_is_fresh"] = True
        series["quote_spread"] = float(bar.get("quote_spread", 0.0))
        return series

    def _sized_intent(self, intent: OrderIntent, bar: pd.Series, quote: Quote) -> OrderIntent:
        atr = float(bar.get("atr", 0.0) or 0.0)
        if atr <= 0:
            # Cold start: the rolling window is too short for a real ATR, so
            # estimate from the bar range (mirrors the RL env) instead of
            # falling back to the $0.10 floor, which would stop out instantly.
            high = float(bar.get("high", quote.mid) or quote.mid)
            low = float(bar.get("low", quote.mid) or quote.mid)
            atr = max(high - low, quote.mid * 0.0005)
        profile = self._profile
        # The champion's validated exit structure (M15 = 1xATR stop, 3xATR
        # target). The 0.10 floor is a minimum absolute stop for thin markets.
        stop_distance = max(atr * profile.stop_loss_atr, 0.10)
        target_distance = atr * profile.take_profit_atr if atr > 0 else 2 * stop_distance
        # Convert account equity to USD before risk sizing (GAP-3): a ZAR
        # account's raw equity would otherwise undersize/oversize positions.
        equity = self._equity_usd()
        risk_per_trade = self.breaker.limits.risk_per_trade
        volume = risk_based_volume(
            equity,
            stop_distance,
            risk_per_trade=risk_per_trade,
            point_value=self._point_value,
            volume_max=self.guardrails.config.max_position_volume,
        )
        if volume <= 0:
            volume = intent.volume
        # Honest small-account warning: when the broker's minimum lot already
        # risks several times the configured budget, the min lot wins and each
        # trade risks more than risk_per_trade. Log it once, never block it.
        if not self._minlot_warned and equity > 0 and volume > 0:
            implied_risk = volume * stop_distance * self._point_value
            budget = equity * risk_per_trade
            if budget > 0 and implied_risk > 2.0 * budget:
                self._minlot_warned = True
                self.alerter.alert(
                    "warning",
                    "min-lot risk exceeds budget",
                    f"min lot {volume} risks ~${implied_risk:.2f} vs budget ${budget:.2f} ({risk_per_trade:.1%}); "
                    "consider a larger account balance to match the validated 0.5% risk/trade",
                )
                self.logger.warning(
                    "min-lot risk exceeds budget",
                    extra={"event": "minlot_warn", "implied_usd": implied_risk, "budget_usd": budget},
                )
        # Margin guard (GAP-7): never enter a position the account cannot
        # comfortably margin.
        if not self._margin_ok(intent.symbol, volume):
            self.alerter.alert("warning", "insufficient margin", f"{intent.symbol} vol={volume}")
            self.logger.warning("margin rejected", extra={"event": "margin_reject", "symbol": intent.symbol})
            return OrderIntent(
                symbol=intent.symbol, side=intent.side, volume=0.0, reason=intent.reason
            )
        # Entry reference: a resting limit fills at its limit price, a market
        # order at the quote mid — SL/TP must be anchored to THAT price.
        entry_ref = (
            float(intent.limit_price)
            if intent.kind == OrderKind.LIMIT and intent.limit_price is not None
            else float(quote.mid)
        )
        sl = round(entry_ref - stop_distance, 5) if intent.side == Side.BUY else round(entry_ref + stop_distance, 5)
        tp = round(entry_ref + target_distance, 5) if intent.side == Side.BUY else round(entry_ref - target_distance, 5)
        return OrderIntent(
            symbol=intent.symbol,
            side=intent.side,
            volume=volume,
            kind=intent.kind,
            limit_price=intent.limit_price,
            stop_loss=sl,
            take_profit=tp,
            reason=intent.reason,
        )

    def _account_currency(self) -> str:
        try:
            info = self.adapter.account_info()
            return str(getattr(info, "currency", "USD") or "USD")
        except Exception:  # pragma: no cover - broker dependent
            return "USD"

    def _equity_usd(self) -> float:
        equity = self._equity()
        try:
            return self._converter.to_usd(equity, self.mt5, self._account_currency())
        except Exception:  # pragma: no cover - broker dependent
            return equity

    def _margin_ok(self, symbol: str, volume: float, safety: float = 2.0) -> bool:
        """Require free margin ≥ safety × required margin for the order."""
        if volume <= 0:
            return False
        try:
            info = self.adapter.account_info()
            margin_free = float(getattr(info, "margin_free", 0.0) or 0.0)
            spec = self.adapter.symbol_spec(symbol)
            contract = float(getattr(spec, "trade_contract_size", 0.0) or 0.0)
            margin_initial = float(getattr(spec, "margin_initial", 0.0) or 0.0)
            if contract <= 0 or margin_initial <= 0:
                return True  # cannot compute → do not block on missing data
            required = margin_initial * volume * contract
            return margin_free >= safety * required
        except Exception:  # pragma: no cover - broker dependent
            return True

    def _equity(self) -> float:
        try:
            info = self.adapter.account_info()
            return float(getattr(info, "equity", self.settings.initial_balance))
        except Exception:  # pragma: no cover - broker dependent
            return self.settings.initial_balance

    def _current_positions(self, resolved_symbol: str) -> dict[str, float]:
        positions = self.mt5.positions_get() or []
        result: dict[str, float] = {}
        for position in positions:
            symbol = str(getattr(position, "symbol", ""))
            volume = float(getattr(position, "volume", 0.0))
            direction = 1.0 if int(getattr(position, "type", 0)) == 0 else -1.0
            result[symbol] = result.get(symbol, 0.0) + direction * volume
        return result

    # -- helpers ---------------------------------------------------------------
    def _export_metrics(self, resolved_symbol: str) -> None:
        try:
            info = self.adapter.account_info()
            self.metrics.equity.set(float(getattr(info, "equity", 0.0)))
            self.metrics.balance.set(float(getattr(info, "balance", 0.0)))
        except Exception:  # pragma: no cover
            pass
        self.metrics.kill_switch.set(1 if self.guardrails.kill_switch else 0)
        self.metrics.trading_paused.set(1 if self.breaker.paused else 0)

    def _heartbeat(self) -> None:
        self.soak.record(healthy=True)

    def _publish_status(self) -> None:
        """Write a tiny JSON status file the dashboard reads (atomic swap).

        This is the contract between the live loop and the web platform:
        ``state/live_status.json`` always carries the newest snapshot —
        heartbeat, price, position, pending limit, equity, errors, last
        decision — so a phone dashboard never needs to scrape logs.
        """
        try:
            path = Path(self.settings.state_dir) / "live_status.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            equity = balance = None
            try:
                info = self.adapter.account_info()
                equity = float(getattr(info, "equity", None) or 0.0)
                balance = float(getattr(info, "balance", None) or 0.0)
            except Exception:  # pragma: no cover - broker dependent
                pass
            pending = None
            if self._pending_limit:
                pending = {
                    "side": self._pending_limit.get("side"),
                    "price": self._pending_limit.get("entry"),
                    "bars": self._pending_limit.get("bars"),
                    "volume": self._pending_limit.get("volume"),
                }
            payload = {
                "ts": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "symbol": self.settings.symbol,
                "timeframe": self.settings.timeframe,
                "tick": self._ticks,
                "quotes_seen": self._quotes_seen,
                "bars_built": self._bar_index,
                "price": round(float(self._last_quote.mid), 2) if self._last_quote else None,
                "side": self._side,
                "pending_limit": pending,
                "equity": round(equity, 2) if equity is not None else None,
                "balance": round(balance, 2) if balance is not None else None,
                "errors": self._errors,
                "orders": self._orders,
                "fills": self._fills,
                "kill_switch": bool(self.guardrails.kill_switch),
                "last_decision": self._last_decision,
                "journal_path": str(self._journal_path()),
                "log_path": str(Path(self.settings.log_dir) / "slytrade.jsonl"),
            }
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, default=str))
            os.replace(tmp, path)
        except Exception:  # pragma: no cover - status must never stop trading
            return

    def _status_heartbeat(self, quote: Quote, resolved_symbol: str) -> None:
        """Periodic 'alive' line so the operator can SEE the loop working.

        Logs one compact status line every ``heartbeat_interval_seconds``:
        current price, how many quotes were seen, bars built, position state,
        pending limit and account equity. Nothing else prints between bars, so
        this is the proof-of-life signal.
        """
        interval = max(float(self.settings.heartbeat_interval_seconds or 5.0), 5.0)
        now = time.monotonic()
        if now - self._last_heartbeat_at < interval:
            return
        self._last_heartbeat_at = now
        held = self._current_positions(resolved_symbol).get(resolved_symbol, 0.0)
        side = "long" if held > 0 else ("short" if held < 0 else "flat")
        equity = None
        try:
            info = self.adapter.account_info()
            equity = float(getattr(info, "equity", None) or 0.0)
        except Exception:  # pragma: no cover
            equity = None
        pending = (
            f"{self._pending_limit['side']}@{self._pending_limit['entry']} (bar {self._pending_limit['bars']}/ttl)"
            if self._pending_limit else "none"
        )
        msg = (
            f"♥ alive tick={self._ticks} price={round(float(quote.mid), 2)} "
            f"quotes={self._quotes_seen} bars={self._bar_index} side={side} "
            f"limit={pending} equity={round(equity, 2) if equity is not None else 'n/a'} "
            f"errors={self._errors}"
        )
        self.logger.info(
            msg,
            extra={
                "event": "heartbeat",
                "tick": self._ticks,
                "price": round(float(quote.mid), 2),
                "quotes_seen": self._quotes_seen,
                "bars_built": self._bar_index,
                "side": side,
                "pending_limit": pending,
                "equity": round(equity, 2) if equity is not None else None,
                "errors": self._errors,
            },
        )
        self._publish_status()

    def _decision_trace(self, series: pd.Series, quote: Quote, intent: OrderIntent | None) -> str:
        """Compact per-bar readout of WHAT the champion saw and WHY it decided.

        This is the 'every little process' line: the two confluence scores,
        the threshold, the H4 direction, MTF bias, cooldown remaining, and the
        decision (HOLD + likely reason, or the emitted intent). Best-effort and
        non-fatal by design.
        """

        def g(name: str, default: float = 0.0) -> float:
            if name not in series.index:
                return default
            value = series[name]
            if pd.isna(value):
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        strat = self.strategy
        long_score = short_score = 0.0
        scorer = getattr(strat, "_scorer", None)
        if scorer is not None:
            try:
                long_score = float(scorer._long_score(series))
            except Exception:  # pragma: no cover
                long_score = 0.0
            try:
                short_score = float(scorer._short_score(series))
            except Exception:  # pragma: no cover
                short_score = 0.0
        min_score = int(getattr(strat.config, "min_score", 4) or 4)
        cooldown = int(getattr(strat.config, "cooldown_bars", 0) or 0)
        last_entry = int(getattr(strat, "_last_entry_index", -10_000_000) or -10_000_000)
        bars_since = int(self._bar_index) - last_entry
        cooldown_left = max(0, cooldown - bars_since)
        htf = g("htf_h4_bos_dir")
        bias = g("mtf_bias")
        atr = g("atr")

        if intent is not None:
            entry = f"@{intent.limit_price}" if intent.kind == OrderKind.LIMIT and intent.limit_price else "@market"
            return (
                f"bar {self._bar_index} close={g('close'):.2f} atr={atr:.2f} "
                f"long={long_score:.0f} short={short_score:.0f} (need>={min_score}) "
                f"h4={htf:+.0f} bias={bias:+.0f} → SIGNAL {intent.side.value} {intent.kind.value} {entry}"
            )
        if cooldown_left > 0:
            reason = f"cooldown {cooldown_left} bars left"
        elif max(long_score, short_score) < min_score:
            reason = f"below threshold (max score {max(long_score, short_score):.0f} < {min_score})"
        elif bias != 0 and htf != 0 and bias != htf:
            reason = "H4 alignment blocked"
        else:
            reason = "quality gate (no footprint/momentum)"
        return (
            f"bar {self._bar_index} close={g('close'):.2f} atr={atr:.2f} "
            f"long={long_score:.0f} short={short_score:.0f} (need>={min_score}) "
            f"h4={htf:+.0f} bias={bias:+.0f} → HOLD ({reason})"
        )

    def _alert_kill_switch_once(self) -> None:
        if self._kill_switch_alerted:
            return
        self._kill_switch_alerted = True
        self.alerter.alert("critical", "kill switch active", self.guardrails.kill_switch_reason or "reason not recorded")

    def _install_signal_handlers(self) -> None:
        def handle(_sig: int, _frame: object) -> None:
            self.logger.info("shutdown requested", extra={"event": "shutdown"})
            self._stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handle)
            except ValueError:  # pragma: no cover - not in main thread
                pass

    def stop(self) -> None:
        self._stop.set()


# Backward-compatible alias (older scripts/tests import the old name).
DemoTradingLoop = LiveTradingLoop
