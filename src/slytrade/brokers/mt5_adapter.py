from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from slytrade.backtest.execution import Quote
from slytrade.brokers.specs import SymbolSpec, symbol_spec_from_mt5_info
from slytrade.execution.models import ExecutionReport, OrderIntent, OrderStatus, Side
from slytrade.execution.oms import OrderManagementSystem
from slytrade.monitoring.health import HealthRegistry
from slytrade.monitoring.metrics import ExecutionMetrics
from slytrade.risk.guardrails import TradingGuardrails


@dataclass(frozen=True)
class ReconciliationResult:
    reconciled: bool
    broker_positions: int
    broker_orders: int
    local_open_orders: int
    detail: str


class MT5BrokerAdapter:
    """Guarded adapter for the official MetaTrader5 Python API.

    The adapter refuses new exposure until an explicit reconciliation succeeds.
    It is also disabled by default; enabling it requires both configuration and
    an explicit deployment gate outside this class.
    """

    def __init__(
        self,
        mt5: Any,
        *,
        oms: OrderManagementSystem,
        guardrails: TradingGuardrails,
        health: HealthRegistry | None = None,
        metrics: ExecutionMetrics | None = None,
        allow_trading: bool = False,
        magic: int = 260810,
        expected_positions: dict[str, float] | None = None,
    ):
        self.mt5 = mt5
        self.oms = oms
        self.guardrails = guardrails
        self.health = health or HealthRegistry()
        self.metrics = metrics or ExecutionMetrics()
        self.allow_trading = allow_trading
        self.magic = magic
        self.expected_positions = expected_positions or {}
        self.reconciled = False

    def connect(self) -> None:
        try:
            initialized = self.mt5.initialize()
        except Exception as exc:
            self.health.report("mt5", False, f"initialize failed: {exc}")
            raise RuntimeError(f"MT5 initialize failed: {exc}") from exc
        if initialized is False:
            self.health.report("mt5", False, "initialize returned False")
            raise RuntimeError("MT5 initialize returned False")
        self.health.report("mt5", True, "connected")

    def disconnect(self) -> None:
        self.reconciled = False
        shutdown = getattr(self.mt5, "shutdown", None)
        if shutdown is not None:
            shutdown()
        self.health.report("mt5", False, "disconnected")

    def account_info(self) -> Any:
        try:
            info = self._call_remote("account_info", "mt5.account_info()._asdict()")
        except Exception as exc:
            self.health.report("account", False, f"account_info failed: {exc}")
            raise RuntimeError(f"MT5 account_info failed; terminal may not be logged in: {exc}") from exc
        if info is None:
            self.health.report("mt5", False, "account_info returned None")
            raise RuntimeError("MT5 account_info returned None; terminal may not be logged in")
        return info

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        info = self._call_remote("symbol_info", f"mt5.symbol_info({json.dumps(symbol)})._asdict()", symbol)
        return symbol_spec_from_mt5_info(info)

    def resolve_symbol(self, requested: str) -> str:
        # The bridge may return SymbolInfo objects, dicts, or name strings
        # depending on which path ``_call_remote`` took — normalise to names.
        raw = self._call_remote("symbols_get", "[s.name for s in (mt5.symbols_get() or [])]")
        names: list[str] = []
        for item in raw:
            name = _field(item, "name")
            names.append(str(name) if name else str(item))
        requested_lower = requested.strip().lower()

        def _rank(name: str) -> tuple[object, ...]:
            lower = name.lower()
            return (
                lower != requested_lower,  # exact match first
                "247" in lower,  # 24/7 weekend contract last (XAUUSDm over XAUUSD247m)
                len(name),
                name,
            )

        candidates = sorted((n for n in names if requested_lower in n.lower()), key=_rank)
        if not candidates:
            raise RuntimeError(f"no MT5 symbol matches {requested}; available symbols: {sorted(names)[:20]}")
        resolved = candidates[0]
        # symbol_select can return False when the symbol can't be added to the
        # Market Watch (e.g. the 24/7 contract, or a full watchlist) — the
        # symbol is still quotable and tradable via order_send, so degrade to a
        # warning instead of a fatal error.
        try:
            selected = self._call_remote(
                "symbol_select",
                f"mt5.symbol_select({json.dumps(resolved)}, True)",
                resolved,
                True,
            )
        except Exception:  # pragma: no cover - bridge dependent
            selected = False
        if selected is False:
            self.health.report(
                "mt5",
                True,
                f"symbol_select({resolved}) returned False (not visible in Market Watch; still tradable)",
            )
        return resolved

    def quote(self, symbol: str) -> Quote:
        tick = self._call_remote("symbol_info_tick", f"mt5.symbol_info_tick({json.dumps(symbol)})._asdict()", symbol)
        if tick is None:
            raise RuntimeError(f"no MT5 quote available for {symbol}; verify the symbol is selected and market is open")
        bid = float(_field(tick, "bid"))
        ask = float(_field(tick, "ask"))
        if bid <= 0 or ask <= 0 or ask < bid:
            raise RuntimeError(f"invalid MT5 quote for {symbol}: bid={bid}, ask={ask}")
        tick_time = _field(tick, "time", 0)
        timestamp = datetime.fromtimestamp(float(tick_time), tz=UTC) if tick_time else datetime.now(UTC)
        return Quote(symbol=symbol, bid=bid, ask=ask, time=timestamp)

    def reconcile(self) -> ReconciliationResult:
        broker_positions = tuple(self._call_remote("positions_get", "[item._asdict() for item in (mt5.positions_get() or [])]") or ())
        broker_orders = tuple(self._call_remote("orders_get", "[item._asdict() for item in (mt5.orders_get() or [])]") or ())
        local_open = self.oms.open_orders()
        broker_ids = {str(_field(order, "ticket", "")) for order in broker_orders}
        known_ids = {state.broker_order_id for state in local_open if state.broker_order_id}
        unknown_broker_orders = broker_ids - known_ids
        actual_positions: dict[str, float] = {}
        for position in broker_positions:
            symbol = str(_field(position, "symbol", ""))
            volume = float(_field(position, "volume", 0.0))
            direction = 1.0 if int(_field(position, "type", 0)) == 0 else -1.0
            actual_positions[symbol] = actual_positions.get(symbol, 0.0) + direction * volume
        positions_match = actual_positions == self.expected_positions
        self.reconciled = not unknown_broker_orders and positions_match
        if self.reconciled:
            detail = "reconciled"
        elif unknown_broker_orders:
            detail = f"unknown broker orders: {sorted(unknown_broker_orders)}"
        else:
            detail = f"position mismatch: broker={actual_positions}, expected={self.expected_positions}"
        result = ReconciliationResult(
            reconciled=self.reconciled,
            broker_positions=len(broker_positions),
            broker_orders=len(broker_orders),
            local_open_orders=len(local_open),
            detail=detail,
        )
        self.health.report("reconciliation", result.reconciled, result.detail)
        return result

    def submit(self, intent: OrderIntent, quote: Quote) -> ExecutionReport:
        if self.oms.get(intent.client_order_id) is not None:
            state = self.oms.get(intent.client_order_id)
            assert state is not None
            return ExecutionReport(
                client_order_id=intent.client_order_id,
                status=state.status,
                filled_volume=state.filled_volume,
                avg_fill_price=state.avg_fill_price,
                broker_order_id=state.broker_order_id,
                message="idempotent existing order",
                event_time=datetime.now(UTC),
            )
        if not self.allow_trading:
            return self._reject(intent, "MT5 trading adapter disabled")
        if not self.reconciled:
            return self._reject(intent, "MT5 reconciliation required before trading")
        if quote.symbol != intent.symbol:
            return self._reject(intent, "quote symbol does not match order")

        equity = float(self.account_info().equity)
        decision = self.guardrails.approve_order(
            intent,
            equity=equity,
            spread_points=quote.spread / max(self.symbol_spec(intent.symbol).point, 1e-12),
            live=True,
            current_date=quote.time.date(),
        )
        if not decision.approved:
            return self._reject(intent, decision.reason)

        self.oms.create_order(intent)
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": intent.symbol,
            "volume": intent.volume,
            "type": getattr(self.mt5, "ORDER_TYPE_BUY" if intent.side == Side.BUY else "ORDER_TYPE_SELL"),
            "price": quote.ask if intent.side == Side.BUY else quote.bid,
            "sl": intent.stop_loss or 0.0,
            "tp": intent.take_profit or 0.0,
            "deviation": 20,
            "magic": self.magic,
            "comment": intent.reason[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        self.metrics.submitted()
        result = self._call_remote("order_send", f"mt5.order_send({json.dumps(request)})._asdict()", request)
        if result is None:
            self.metrics.broker_error()
            return self._reject(intent, "MT5 order_send returned None")
        retcode = int(_field(result, "retcode", -1))
        filled = retcode in {
            int(getattr(self.mt5, "TRADE_RETCODE_DONE", -2)),
            int(getattr(self.mt5, "TRADE_RETCODE_DONE_PARTIAL", -3)),
        }
        status = OrderStatus.FILLED if filled else OrderStatus.REJECTED
        if filled:
            self.metrics.filled()
        else:
            self.metrics.rejected()
        report = ExecutionReport(
            client_order_id=intent.client_order_id,
            status=status,
            filled_volume=float(_field(result, "volume", intent.volume if filled else 0.0)),
            avg_fill_price=float(_field(result, "price", 0.0)) if filled else None,
            broker_order_id=str(_field(result, "order", "")) if _field(result, "order", 0) else None,
            message=str(_field(result, "comment", "")) or f"MT5 retcode {retcode}",
            event_time=quote.time,
        )
        self.oms.apply_report(report)
        return report

    def _reject(self, intent: OrderIntent, message: str) -> ExecutionReport:
        self.metrics.rejected()
        self.oms.create_order(intent)
        report = ExecutionReport(client_order_id=intent.client_order_id, status=OrderStatus.REJECTED, message=message)
        self.oms.apply_report(report)
        return report

    def _call_remote(self, method: str, expression: str, *args: object) -> Any:
        """Call an MT5 method, falling back to bridge-side dict conversion."""
        try:
            return getattr(self.mt5, method)(*args)
        except Exception as exc:
            container = getattr(self.mt5, "_container", None)
            if container is None or "pickle" not in str(exc).lower():
                raise
            result = container.eval(expression)
            return _namespace(result)


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _field(value: Any, name: str, default: object = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
