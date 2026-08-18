"""Paper-trading runtime loop.

This is the missing "production runtime": it drives the exact same guarded
execution path as the backtests, but on a live or replayed quote stream.

    QuoteStream -> BarBuilder -> ICT feature engine -> Strategy -> Guardrails
                 -> OMS -> PaperBroker -> Portfolio -> Ledger -> Persistence
                                                                 -> Metrics
                                                                 -> Soak monitor

The loop is fail-closed:

* it refuses to start if :meth:`RuntimeSettings.fail_closed_checks` reports a
  problem;
* new entries are gated by the trading window and the loss circuit breaker;
* risk-reducing exits are never blocked (kill switch / breaker pause entries
  only);
* OMS + ledger state is journaled to SQLite and rehydrated on restart;
* every heartbeat is recorded in a :class:`SoakMonitor` and exported as
  Prometheus metrics.
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import pandas as pd

from slytrade.backtest.execution import ExecutionConfig, Quote
from slytrade.backtest.trade_management import (
    ManagedTradeState,
    TradeManagementConfig,
    create_trade_state,
    next_exit_event,
    quote_for_exit_price,
    update_trailing_stop,
)
from slytrade.core.config import load_config
from slytrade.currency import CurrencyConverter
from slytrade.data.timeframes import timeframe_duration
from slytrade.execution.journal import SqliteJournal
from slytrade.execution.ledger import TradeLedger
from slytrade.execution.models import OrderIntent, OrderStatus
from slytrade.execution.oms import OrderManagementSystem
from slytrade.execution.paper_broker import PaperBroker
from slytrade.features.ict import compute_ict_features
from slytrade.monitoring.gates import DeploymentStage
from slytrade.monitoring.operations import SoakMonitor
from slytrade.risk.guardrails import GuardrailConfig, TradingGuardrails
from slytrade.risk.sizing import risk_based_volume
from slytrade.runtime.alerting import AlertManager
from slytrade.runtime.circuit_breaker import LossCircuitBreaker, limits_from_config
from slytrade.runtime.logs import setup_logging
from slytrade.runtime.metrics_server import TradingMetrics
from slytrade.runtime.news_gate import NewsGate
from slytrade.runtime.settings import RuntimeSettings
from slytrade.runtime.trading_window import TradingWindow, window_from_settings
from slytrade.strategies.personality_adaptive import PersonalityAdaptiveStrategy


class QuoteProvider(Protocol):
    """Source of executable bid/ask quotes."""

    symbol: str
    realtime: bool = False

    def connect(self) -> None: ...
    def next_quote(self) -> Quote | None: ...
    def disconnect(self) -> None: ...


class ReplayQuoteProvider:
    """Replay ticks from a canonical tick file (CSV/Parquet) for paper replay."""

    realtime = False

    def __init__(self, ticks: pd.DataFrame, symbol: str | None = None) -> None:
        required = {"time_msc", "bid", "ask"}
        missing = required.difference(ticks.columns)
        if missing:
            raise ValueError(f"ticks missing required columns: {sorted(missing)}")
        frame = ticks.sort_values("time_msc").reset_index(drop=True)
        if symbol is None:
            symbols = sorted(str(value) for value in frame["symbol"].dropna().unique()) if "symbol" in frame.columns else []
            symbol = symbols[0] if len(symbols) == 1 else "XAUUSD"
        self.symbol = symbol
        self._ticks = frame
        self._index = 0

    def connect(self) -> None:
        self._index = 0

    def next_quote(self) -> Quote | None:
        if self._index >= len(self._ticks):
            return None
        row = self._ticks.iloc[self._index]
        self._index += 1
        return Quote(
            symbol=self.symbol,
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            time=pd.Timestamp(row["time_msc"]).to_pydatetime(),
        )

    def disconnect(self) -> None:
        return None


class MT5QuoteProvider:
    """Live quote provider backed by the MT5 bridge (native or mt5linux)."""

    realtime = True

    def __init__(self, symbol: str, mt5: Any, *, poll_seconds: float = 1.0) -> None:
        self.symbol = symbol
        self.mt5 = mt5
        self.poll_seconds = poll_seconds
        self.resolved_symbol = symbol

    def connect(self) -> None:
        if hasattr(self.mt5, "initialize"):
            ok = self.mt5.initialize()
            if ok is False:
                raise RuntimeError("MT5 initialize() returned False")
        # Resolve base symbol -> broker-specific symbol (e.g. XAUUSD -> XAUUSDm).
        if hasattr(self.mt5, "symbols_get"):
            names = [str(s.name) for s in (self.mt5.symbols_get() or [])]
            lowered = self.symbol.strip().lower()
            candidates = sorted(
                (n for n in names if lowered in str(n).lower()),
                key=lambda n: (str(n).lower() != lowered, len(str(n)), str(n)),
            )
            if candidates:
                self.resolved_symbol = str(candidates[0])
                if hasattr(self.mt5, "symbol_select"):
                    self.mt5.symbol_select(self.resolved_symbol, True)

    def next_quote(self) -> Quote | None:
        tick = self.mt5.symbol_info_tick(self.resolved_symbol)
        if tick is None:
            return None
        bid = float(getattr(tick, "bid", 0.0))
        ask = float(getattr(tick, "ask", 0.0))
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        tick_time = getattr(tick, "time", 0)
        timestamp = datetime.fromtimestamp(float(tick_time), tz=UTC) if tick_time else datetime.now(UTC)
        return Quote(symbol=self.resolved_symbol, bid=bid, ask=ask, time=timestamp)

    def disconnect(self) -> None:
        if hasattr(self.mt5, "shutdown"):
            self.mt5.shutdown()


class BarBuilder:
    """Aggregate ticks into timeframe bars (mid-based OHLC + last quote)."""

    def __init__(self, symbol: str, timeframe: str = "M1") -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self._current: dict[str, Any] | None = None

    def _bucket(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.floor(pd.Timedelta(timeframe_duration(self.timeframe)))

    def seed(self, bar: dict[str, Any]) -> None:
        """Seed the in-progress bar so the builder CONTINUES it.

        Used at startup after a broker warmup: the terminal already holds the
        current (still-forming) bar, so the builder joins it mid-bar instead of
        restarting from scratch. The next completed bar is the natural close of
        the seeded bucket — the bot is immediately in sync with the market.
        """
        seeded = dict(bar)
        if "quote_bid" not in seeded or not seeded.get("quote_bid"):
            seeded["quote_bid"] = float(seeded.get("close", 0.0) or 0.0)
        if "quote_ask" not in seeded or not seeded.get("quote_ask"):
            seeded["quote_ask"] = float(seeded.get("close", 0.0) or 0.0)
        if not seeded.get("tick_volume"):
            seeded["tick_volume"] = 1
        seeded.setdefault("quote_time", None)
        self._current = seeded

    def _new_bar(self, bucket: pd.Timestamp, quote: Quote) -> dict[str, Any]:
        return {
            "time": bucket,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open": quote.mid,
            "high": quote.mid,
            "low": quote.mid,
            "close": quote.mid,
            "tick_volume": 1,
            "quote_bid": quote.bid,
            "quote_ask": quote.ask,
            "quote_time": quote.time,
        }

    def on_quote(self, quote: Quote) -> dict[str, Any] | None:
        ts = pd.Timestamp(quote.time)
        bucket = self._bucket(ts)
        completed: dict[str, Any] | None = None
        if self._current is not None:
            current_bucket = pd.Timestamp(self._current["time"])
            if bucket == current_bucket:
                self._current["high"] = max(self._current["high"], quote.mid)
                self._current["low"] = min(self._current["low"], quote.mid)
                self._current["close"] = quote.mid
                self._current["tick_volume"] += 1
                self._current["quote_bid"] = quote.bid
                self._current["quote_ask"] = quote.ask
                self._current["quote_time"] = quote.time
                return None
            completed = self._finalize(self._current)
        self._current = self._new_bar(bucket, quote)
        return completed

    def _finalize(self, bar: dict[str, Any]) -> dict[str, Any]:
        open_ts = pd.Timestamp(bar["time"])
        bar["decision_time"] = open_ts + pd.Timedelta(timeframe_duration(self.timeframe))
        bar["quote_is_fresh"] = True
        bar["quote_spread"] = (bar["quote_ask"] - bar["quote_bid"]) if bar["quote_ask"] >= bar["quote_bid"] else 0.0
        return bar

    def close_current(self) -> dict[str, Any] | None:
        if self._current is None:
            return None
        completed = self._finalize(self._current)
        self._current = None
        return completed


@dataclass(frozen=True)
class LoopSummary:
    bars_processed: int
    orders_submitted: int
    orders_filled: int
    trades_closed: int
    final_equity: float
    kill_switch: bool
    paused: bool
    errors: int


@dataclass
class PaperTradingLoop:
    """Drives the guarded paper-execution path over a quote stream."""

    settings: RuntimeSettings
    provider: QuoteProvider
    strategy: Any = None
    portfolio_breaker: Any = None  # optional shared multi-symbol breaker
    logger: Any = field(default=None, init=False)

    # Runtime state
    broker: PaperBroker = field(init=False)
    guardrails: TradingGuardrails = field(init=False)
    breaker: LossCircuitBreaker = field(init=False)
    window: TradingWindow = field(init=False)
    news_gate: NewsGate = field(init=False)
    alerter: AlertManager = field(init=False)
    metrics: TradingMetrics = field(default_factory=TradingMetrics, init=False)
    soak: SoakMonitor = field(init=False)
    bar_builder: BarBuilder = field(init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _trade_state: ManagedTradeState | None = field(default=None, init=False)
    _window_bars: list[dict[str, Any]] = field(default_factory=list, init=False)
    _trade_realized: float = field(default=0.0, init=False)
    _bar_index: int = field(default=0, init=False)
    _errors: int = field(default=0, init=False)
    _trades_closed_count: int = field(default=0, init=False)
    _kill_switch_alerted: bool = field(default=False, init=False)
    _summary: LoopSummary | None = field(default=None, init=False)

    # -- lifecycle ----------------------------------------------------------
    def __post_init__(self) -> None:
        self.logger = setup_logging(
            self.settings.log_level,
            self.settings.log_dir,
            self.settings.json_logs,
        )
        problems = self.settings.fail_closed_checks()
        if problems:
            raise ValueError("startup blocked: " + "; ".join(problems))

        config = load_config(self.settings.config_dir)
        risk_cfg = config.risk
        limits = limits_from_config(risk_cfg)
        # Account-currency handling: sizing math uses USD point values, so a
        # non-USD account equity must be converted before risk-based sizing.
        # Env settings win; risk.yaml's `currency:` block is the fallback.
        currency_cfg = risk_cfg.get("currency", {}) or {}
        self._account_currency = (
            self.settings.account_currency or str(currency_cfg.get("account_currency", "USD"))
        ).upper()
        rate = float(self.settings.currency_rate_to_usd or 1.0)
        if rate == 1.0 and currency_cfg.get("rate_to_usd"):
            rate = float(currency_cfg["rate_to_usd"])
        self._converter = CurrencyConverter(fallback_rate=rate)
        self.breaker = LossCircuitBreaker(limits)
        self.window = window_from_settings(
            self.settings.trading_days,
            self.settings.trading_start_utc,
            self.settings.trading_end_utc,
        )
        self.alerter = AlertManager.from_settings(self.settings, self.logger)
        from slytrade.runtime.calendar import build_news_gate_from_settings

        self.news_gate = build_news_gate_from_settings(self.settings)

        self.guardrails = TradingGuardrails(
            GuardrailConfig(
                allow_live_trading=self.settings.allow_live,
                max_daily_drawdown=limits.max_daily_drawdown,
                max_total_drawdown=limits.max_total_drawdown,
                max_position_volume=float(risk_cfg.get("max_position_volume", 1.0)),
                max_spread_points=float(config.broker.get("execution", {}).get("max_spread_points", 50.0)),
            ),
            initial_equity=self.settings.initial_balance,
            kill_switch_path=self.settings.kill_switch_path,
        )

        # Durable OMS + ledger (survives restarts).
        self.settings.state_path.mkdir(parents=True, exist_ok=True)
        journal = SqliteJournal(self.settings.state_path / "execution-events.db")
        oms = OrderManagementSystem(journal)
        ledger = TradeLedger(journal)

        self.broker = PaperBroker(
            initial_balance=self.settings.initial_balance,
            execution_config=ExecutionConfig(
                point_size=self._point_size(),
                point_value=self._point_value(),
            ),
            guardrail_config=None,
            oms=oms,
            ledger=ledger,
        )
        # Rebuild the broker's guardrails with our (env-aware) instance.
        self.broker.guardrails = self.guardrails

        self.soak = SoakMonitor(DeploymentStage.PAPER, min_samples=1, stale_after=timedelta(minutes=5))
        self.bar_builder = BarBuilder(self.settings.symbol, self.settings.timeframe)

        if self.strategy is None:
            from slytrade.tasks import _persona_config_from_risk

            self.strategy = PersonalityAdaptiveStrategy(
                symbol=self.settings.symbol,
                volume=0.1,
                config=_persona_config_from_risk(self.settings.symbol, self.settings.timeframe),
            )

    def _point_size(self) -> float:
        try:
            from slytrade.brokers.specs import load_symbol_spec, spec_to_backtest_pricing

            if self.settings.symbol_spec_file:
                return spec_to_backtest_pricing(load_symbol_spec(self.settings.symbol_spec_file)).point_size
        except Exception:  # pragma: no cover - spec file is optional
            pass
        return 0.01

    def _point_value(self) -> float:
        try:
            from slytrade.brokers.specs import load_symbol_spec, spec_to_backtest_pricing

            if self.settings.symbol_spec_file:
                return spec_to_backtest_pricing(load_symbol_spec(self.settings.symbol_spec_file)).point_value
        except Exception:  # pragma: no cover - spec file is optional
            pass
        return 1.0

    def _equity_usd(self) -> float:
        """Current equity converted to USD for sizing math.

        The portfolio marks positions in account currency; point_value is
        USD-based, so risk-budgeted volume must be computed from the
        USD-equivalent equity. USD accounts are a no-op.
        """
        equity = self.broker.portfolio.mark_to_market(self.broker.last_marks)
        if self._account_currency == "USD":
            return equity
        return self._converter.to_usd(equity)

    def _equity_account(self) -> float:
        """Current equity in account currency (for reporting/metrics)."""
        return self.broker.portfolio.mark_to_market(self.broker.last_marks)

    # -- main entry -----------------------------------------------------------
    def run(self, *, max_bars: int | None = None, max_seconds: float = 0.0) -> LoopSummary:
        self._install_signal_handlers()
        start = time.monotonic()
        self.logger.info("paper loop starting", extra={"event": "start", "symbol": self.settings.symbol})
        self.provider.connect()
        self._warmup()
        try:
            while not self._stop.is_set():
                elapsed = time.monotonic() - start
                if max_seconds and elapsed >= max_seconds:
                    break
                if max_bars is not None and self._bar_index >= max_bars:
                    break

                try:
                    quote = self.provider.next_quote()
                except Exception as exc:  # pragma: no cover - broker dependent
                    self._errors += 1
                    self.metrics.broker_errors_total.inc()
                    self.soak.record(healthy=False, detail=f"quote error: {exc}")
                    self.logger.error("quote provider error", exc_info=exc, extra={"event": "quote_error"})
                    self.alerter.alert("warning", "quote provider error", str(exc))
                    if not self.provider.realtime:
                        break
                    time.sleep(self.settings.poll_seconds)
                    continue

                if quote is None:
                    if not self.provider.realtime:
                        break
                    self._heartbeat(healthy=True)
                    time.sleep(self.settings.poll_seconds)
                    continue

                if self._quote_is_stale(quote):
                    self.metrics.stale_quotes_total.inc()
                    self.broker.update_quote(quote)
                    self._heartbeat(healthy=True)
                    time.sleep(self.settings.poll_seconds)
                    continue

                self.broker.update_quote(quote)
                bar = self.bar_builder.on_quote(quote)
                if bar is not None:
                    self._bar_index += 1
                    self._on_bar(bar)
                self._heartbeat(healthy=True)

                if self.provider.realtime:
                    time.sleep(self.settings.poll_seconds)
        finally:
            trailing = self.bar_builder.close_current()
            if trailing is not None:
                self._on_bar(trailing)
            self.provider.disconnect()
            self._export_metrics()

        self._summary = LoopSummary(
            bars_processed=self._bar_index,
            orders_submitted=len(self.broker.oms.orders),
            orders_filled=self._orders_filled(),
            trades_closed=self._trades_closed_count,
            final_equity=self.broker.portfolio.mark_to_market(self.broker.last_marks),
            kill_switch=self.guardrails.kill_switch,
            paused=self.breaker.paused,
            errors=self._errors,
        )
        self.logger.info("paper loop stopped", extra={"event": "stop", "equity": self._summary.final_equity})
        if self.soak.alerts:
            for soak_alert in self.soak.alerts:
                self.alerter.alert("warning", f"soak: {soak_alert.code}", soak_alert.detail)
        self.alerter.alert(
            "info",
            "paper loop stopped",
            f"bars={self._summary.bars_processed} equity={self._summary.final_equity:.2f} errors={self._summary.errors}",
        )
        return self._summary

    # -- per-bar logic ----------------------------------------------------------
    def _on_bar(self, bar: dict[str, Any]) -> None:
        self._window_bars.append(bar)
        cap = max(64, int(self.settings.history_bars))
        if len(self._window_bars) > cap:
            self._window_bars.pop(0)

        decision_bar = self._decision_bar(bar)
        quote = self._quote_from_bar(bar)

        if self._trade_state is not None:
            self._manage_open_trade(decision_bar, quote)
        else:
            self._maybe_enter(decision_bar, quote)

        self._export_metrics()
        self.soak.record(healthy=True)

    def _decision_bar(self, bar: dict[str, Any]) -> pd.Series:
        """Merge the raw bar with causally computed ICT + multi-timeframe features."""
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
        # Multi-timeframe context (H4/D1) so the champion's H4-trend alignment
        # gate — the validated edge — actually runs live instead of silently
        # degrading to a single-timeframe strategy.
        for column, value in self._higher_timeframe_features(frame).items():
            series[column] = value
        duration = max(1.0, timeframe_duration(self.settings.timeframe).total_seconds())
        series["tick_rate_per_second"] = float(bar.get("tick_volume", 1)) / duration
        series["quote_spread"] = float(bar.get("quote_spread", 0.0))
        series["quote_is_fresh"] = True
        return series

    def _higher_timeframe_features(self, frame: pd.DataFrame) -> dict[str, float]:
        """Resample the rolling window to H4/D1 and merge higher-TF ICT features.

        Produces the ``htf_*`` columns plus ``mtf_bias`` / ``mtf_confluence_score``
        that activate the strategy's MTF alignment gate. Returns {} while the
        window is too short to form a single higher-timeframe bar.
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

    def _warmup(self) -> None:
        """Seed the feature window from a stored bars file before streaming.

        Pre-fills ATR/EMA/H4/D1 context so signals are warm from the first
        streamed bar instead of cold for the first ``history_bars``. The file is
        optional: without it the loop still runs (and warms up live).
        """
        path = self.settings.replay_bars_file
        if not path:
            return
        try:
            frame = pd.read_parquet(path)
        except Exception:
            try:
                frame = pd.read_csv(path)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("warmup skipped", extra={"event": "warmup_skip", "reason": str(exc)})
                return
        if frame is None or frame.empty:
            return
        frame = frame.sort_values("time")
        cap = max(64, int(self.settings.history_bars))
        self._window_bars.extend(row.to_dict() for _, row in frame.tail(cap).iterrows())
        self.logger.info("warmup loaded", extra={"event": "warmup", "bars": len(frame.tail(cap))})

    @staticmethod
    def _quote_from_bar(bar: dict[str, Any]) -> Quote:
        return Quote(
            symbol=str(bar["symbol"]),
            bid=float(bar["quote_bid"]),
            ask=float(bar["quote_ask"]),
            time=pd.Timestamp(bar["quote_time"]).to_pydatetime(),
        )

    def _maybe_enter(self, bar: pd.Series, quote: Quote) -> None:
        if self.guardrails.kill_switch:
            self._alert_kill_switch_once()
            return
        if not self.window.is_open(quote.time):
            self.logger.debug("window closed", extra={"event": "window_closed"})
            return
        if self.news_gate.is_red_folder(quote.time):
            self.metrics.news_pauses_total.inc()
            self.metrics.news_paused.set(1)
            self.logger.info(
                "red folder: entries paused",
                extra={"event": "news_pause", "reason": self.news_gate.reason(quote.time) or "news"},
            )
            return
        self.metrics.news_paused.set(0)
        if self.portfolio_breaker is not None and not self.portfolio_breaker.allowed():
            self.logger.info(
                "portfolio breaker tripped",
                extra={"event": "portfolio_pause", "reason": self.portfolio_breaker.reason},
            )
            return
        breaker = self.breaker.check()
        if not breaker.allowed:
            self.logger.info("entry paused", extra={"event": "paused", "reason": breaker.reason})
            return

        intent = self.strategy.on_bar(self._bar_index, bar)
        if intent is None:
            return

        # Consistent risk-budgeted sizing regardless of which strategy emitted it.
        # The position cap from risk.yaml (max_position_volume) always wins so a
        # strategy can never size past the guardrails.
        atr = float(bar.get("atr", 0.0) or 0.0)
        stop_distance = max(atr, 0.10)
        # Sizing is computed on USD-equivalent equity: point_value is USD-based
        # (gold = $100/lot/point), so a ZAR account must be converted first.
        equity = self._equity_usd()
        volume = risk_based_volume(
            equity,
            stop_distance,
            risk_per_trade=self.breaker.limits.risk_per_trade,
            point_value=self._point_value(),
            volume_max=self.guardrails.config.max_position_volume,
        )
        if volume <= 0:
            volume = intent.volume
        sized = OrderIntent(
            symbol=intent.symbol,
            side=intent.side,
            volume=volume,
            kind=intent.kind,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            reason=intent.reason,
        )

        result = self.broker.submit_order(sized, quote)
        self.metrics.orders_total.labels(status=str(result.report.status.value)).inc()
        if result.report.status == OrderStatus.FILLED and result.report.avg_fill_price is not None:
            trade_config = self._trade_management_config()
            self._trade_state = create_trade_state(
                sized,
                result.report.avg_fill_price,
                bar,
                self._bar_index,
                trade_config,
            )
            self._trade_realized = 0.0
            self.logger.info(
                "entry filled",
                extra={"event": "entry", "symbol": sized.symbol, "order_id": sized.client_order_id, "volume": sized.volume},
            )
        elif result.report.status == OrderStatus.REJECTED:
            self.logger.warning("entry rejected", extra={"event": "reject", "reason": result.report.message})

    def _manage_open_trade(self, bar: pd.Series, quote: Quote) -> None:
        trade = self._trade_state
        assert trade is not None
        trade_config = self._trade_management_config()
        update_trailing_stop(trade, bar, trade_config)
        reason, volume, exit_price = next_exit_event(trade, bar, self._bar_index, trade_config)
        if reason == "none" or volume <= 0:
            return
        exit_quote = quote_for_exit_price(trade, exit_price, quote) if exit_price is not None else quote
        exit_intent = OrderIntent(
            symbol=trade.symbol,
            side=trade.exit_side,
            volume=volume,
            reason=f"managed_{reason}",
        )
        result = self.broker.submit_order(exit_intent, exit_quote)
        self.metrics.orders_total.labels(status=str(result.report.status.value)).inc()
        if result.report.filled_volume > 0:
            trade.remaining_volume -= result.report.filled_volume
            # The broker journals realized PnL on the closing fill.
            if self.broker.ledger.records:
                self._trade_realized += float(self.broker.ledger.records[-1].realized_pnl)
            if trade.is_closed:
                self._close_trade(self._trade_realized)
                self._trade_state = None

    def _alert_kill_switch_once(self) -> None:
        if self._kill_switch_alerted:
            return
        self._kill_switch_alerted = True
        self.alerter.alert(
            "critical",
            "kill switch active",
            self.guardrails.kill_switch_reason or "reason not recorded",
        )

    def _close_trade(self, realized: float) -> None:
        outcome = "win" if realized > 0 else ("loss" if realized < 0 else "breakeven")
        self.metrics.trades_total.labels(outcome=outcome).inc()
        self.breaker.record_trade(realized)
        if self.portfolio_breaker is not None:
            self.portfolio_breaker.record(self.settings.symbol, realized)
        self._trades_closed_count += 1
        self.logger.info(
            "trade closed",
            extra={"event": "exit", "realized": round(realized, 4), "outcome": outcome},
        )

    def _trade_management_config(self) -> TradeManagementConfig:
        from slytrade.config.timeframe_profiles import profile_for

        config = load_config(self.settings.config_dir)
        ict = config.risk.get("ict", {})
        profile = profile_for(self.settings.timeframe)
        trailing = ict.get("trailing_atr_mult")
        return TradeManagementConfig(
            stop_loss_atr=float(ict.get("sl_atr_mult", profile.stop_loss_atr)),
            take_profit_atr=float(ict.get("tp_atr_mult", profile.take_profit_atr)),
            trailing_stop_atr=float(trailing) if trailing is not None else None,
            partial_take_profit_enabled=bool(ict.get("partial_tp_enabled", False)),
            partial_take_profit_atr=float(ict.get("partial_tp_atr_mult", 1.0)),
            partial_close_fraction=float(ict.get("partial_close_fraction", 0.5)),
            move_to_breakeven_after_partial=bool(ict.get("move_to_breakeven_after_partial", True)),
            trail_after_partial=bool(ict.get("trail_after_partial", True)),
            breakeven_at_r=(float(ict["breakeven_at_r"]) if ict.get("breakeven_at_r") is not None else None),
            max_bars_in_trade=(
                int(ict["max_bars_in_trade"]) if ict.get("max_bars_in_trade") is not None else profile.max_bars_in_trade
            ),
        )

    # -- helpers ---------------------------------------------------------------
    def _quote_is_stale(self, quote: Quote) -> bool:
        if not self.provider.realtime:
            return False
        age = (datetime.now(UTC) - quote.time).total_seconds()
        return age > self.settings.stale_quote_seconds

    def _heartbeat(self, *, healthy: bool) -> None:
        self.soak.record(healthy=healthy)

    def _export_metrics(self) -> None:
        equity = self.broker.portfolio.mark_to_market(self.broker.last_marks)
        self.metrics.equity.set(equity)
        self.metrics.balance.set(self.broker.portfolio.balance or 0.0)
        self.metrics.open_positions.set(
            sum(0 if position.is_flat else 1 for position in self.broker.portfolio.positions.values())
        )
        peak = self.guardrails.peak_equity
        session_start = self.guardrails.session_start_equity
        self.metrics.total_drawdown.set(max(0.0, (peak - equity) / max(peak, 1e-9)))
        self.metrics.daily_drawdown.set(max(0.0, (session_start - equity) / max(session_start, 1e-9)))
        self.metrics.kill_switch.set(1 if self.guardrails.kill_switch else 0)
        self.metrics.trading_paused.set(1 if self.breaker.paused else 0)

    def _install_signal_handlers(self) -> None:
        def handle(_sig: int, _frame: object) -> None:
            self.logger.info("shutdown requested", extra={"event": "shutdown"})
            self._stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handle)
            except ValueError:  # pragma: no cover - not in main thread
                pass

    def _orders_filled(self) -> int:
        return sum(1 for state in self.broker.oms.orders.values() if state.status == OrderStatus.FILLED)

    def stop(self) -> None:
        self._stop.set()
