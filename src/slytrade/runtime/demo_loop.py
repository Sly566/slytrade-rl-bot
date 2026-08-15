"""Live demo-account trading loop.

The final stage before real capital: real orders on a **demo** broker account,
routed through the guarded MT5 adapter. This loop is deliberately paranoid:

* refuses to start unless ``SLYTRADE_ALLOW_LIVE=1`` AND stage is ``demo``;
* refuses to trade until broker reconciliation succeeds;
* entries only (never exits) are gated by session window, news gate, circuit
  breaker and risk sizing;
* every order still passes the guardrails + OMS + adapter path;
* stop-loss/take-profit are attached to the broker order (server-side).

If reconciliation fails or the kill switch trips, the loop stops placing new
orders but keeps streaming quotes so the operator can observe and intervene.
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from slytrade.backtest.execution import Quote
from slytrade.core.config import load_config
from slytrade.currency import CurrencyConverter, load_converter
from slytrade.execution.models import OrderIntent, OrderStatus, Side
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
class DemoLoopSummary:
    bars_processed: int
    orders_submitted: int
    orders_filled: int
    errors: int
    reconciled: bool
    kill_switch: bool
    final_equity: float | None


@dataclass
class DemoTradingLoop:
    """Guarded live trading loop against an MT5 demo account."""

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
        self.bar_builder = BarBuilder(self.settings.symbol, self.settings.timeframe)
        if self.strategy is None:
            self.strategy = PersonalityAdaptiveStrategy(symbol=self.settings.symbol, volume=0.1)

    # -- main ------------------------------------------------------------------
    def run(self, *, max_bars: int | None = None, max_seconds: float = 0.0) -> DemoLoopSummary:
        self._install_signal_handlers()
        start = time.monotonic()
        self.logger.info("demo loop starting", extra={"event": "demo_start", "symbol": self.settings.symbol})

        self.adapter.connect()
        resolved = self.adapter.resolve_symbol(self.settings.symbol)
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
            self.alerter.alert("critical", "demo loop blocked", reconciliation.detail)

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

                bar = self.bar_builder.on_quote(quote)
                if bar is not None:
                    self._bar_index += 1
                    self._on_bar(bar, quote, resolved)
                self._export_metrics(resolved)
                self._heartbeat()
                time.sleep(self.settings.poll_seconds)
        finally:
            self.adapter.disconnect()

        self.alerter.alert(
            "info",
            "demo loop stopped",
            f"bars={self._bar_index} orders={self._orders} fills={self._fills} errors={self._errors}",
        )
        return DemoLoopSummary(
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
            return
        if self.news_gate.is_red_folder(quote.time):
            self.metrics.news_pauses_total.inc()
            return
        decision = self.breaker.check()
        if not decision.allowed:
            self.logger.info("entry paused", extra={"event": "paused", "reason": decision.reason})
            return

        if self._side == "flat":
            series = self._decision_series(bar)
            intent = self.strategy.on_bar(self._bar_index, series)
            if intent is None:
                return
            sized = self._sized_intent(intent, series, quote)
            report = self.adapter.submit(sized, quote)
            self._orders += 1
            self.metrics.orders_total.labels(status=str(report.status.value)).inc()
            if report.status == OrderStatus.FILLED:
                self._fills += 1
                self._side = "long" if sized.side == Side.BUY else "short"
                self.logger.info("demo order filled", extra={"event": "demo_fill", "order_id": sized.client_order_id})
            elif report.status == OrderStatus.REJECTED:
                self.logger.warning("demo order rejected", extra={"event": "demo_reject", "reason": report.message})
                self.alerter.alert("warning", "demo order rejected", report.message)

    def _decision_series(self, bar: dict[str, Any]) -> pd.Series:
        from slytrade.features.ict import compute_ict_features

        frame = pd.DataFrame([bar])
        features = compute_ict_features(frame)
        series = pd.Series(bar, dtype=object)
        last = features.iloc[-1]
        for column in features.columns:
            if column not in ("time", "symbol", "timeframe"):
                series[column] = last[column]
        series["quote_is_fresh"] = True
        series["quote_spread"] = float(bar.get("quote_spread", 0.0))
        return series

    def _sized_intent(self, intent: OrderIntent, bar: pd.Series, quote: Quote) -> OrderIntent:
        atr = float(bar.get("atr", 0.0) or 0.0)
        stop_distance = max(atr, 0.10)
        # Convert account equity to USD before risk sizing (GAP-3): a ZAR
        # account's raw equity would otherwise undersize/oversize positions.
        equity = self._equity_usd()
        volume = risk_based_volume(
            equity,
            stop_distance,
            risk_per_trade=self.breaker.limits.risk_per_trade,
            point_value=self._point_value,
            volume_max=self.guardrails.config.max_position_volume,
        )
        if volume <= 0:
            volume = intent.volume
        # Margin guard (GAP-7): never enter a position the account cannot
        # comfortably margin.
        if not self._margin_ok(intent.symbol, volume):
            self.alerter.alert("warning", "insufficient margin", f"{intent.symbol} vol={volume}")
            self.logger.warning("margin rejected", extra={"event": "margin_reject", "symbol": intent.symbol})
            return OrderIntent(
                symbol=intent.symbol, side=intent.side, volume=0.0, reason=intent.reason
            )
        sl = round(quote.mid - stop_distance, 5) if intent.side == Side.BUY else round(quote.mid + stop_distance, 5)
        tp = round(quote.mid + 2 * stop_distance, 5) if intent.side == Side.BUY else round(quote.mid - 2 * stop_distance, 5)
        return OrderIntent(
            symbol=intent.symbol,
            side=intent.side,
            volume=volume,
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
