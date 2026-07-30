from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from slytrade.execution.models import ExecutionReport, OrderIntent, OrderKind, OrderStatus, Side


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    time: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class ExecutionConfig:
    point_size: float = 0.01
    point_value: float = 1.0
    slippage_points: float = 0.0
    commission_per_volume: float = 0.0
    reject_crossed_spread: bool = True


@dataclass(frozen=True)
class SimulatedFill:
    report: ExecutionReport
    commission: float
    point_value: float


class TickExecutionSimulator:
    """Simple tick/quote execution simulator.

    Market orders fill at ask for buys and bid for sells, with adverse slippage.
    Limit orders only fill when the quote crosses the limit price.
    """

    def __init__(self, config: ExecutionConfig | None = None):
        self.config = config or ExecutionConfig()

    def execute(self, intent: OrderIntent, quote: Quote) -> SimulatedFill:
        if intent.symbol != quote.symbol:
            return self._reject(intent, "quote symbol does not match order symbol")
        if intent.volume <= 0:
            return self._reject(intent, "order volume must be positive")
        if quote.bid <= 0 or quote.ask <= 0:
            return self._reject(intent, "quote prices must be positive")
        if self.config.reject_crossed_spread and quote.ask < quote.bid:
            return self._reject(intent, "crossed spread")

        fill_price: float | None = None
        if intent.kind == OrderKind.MARKET:
            fill_price = quote.ask if intent.side == Side.BUY else quote.bid
        elif intent.kind == OrderKind.LIMIT:
            if intent.limit_price is None:
                return self._reject(intent, "limit order missing limit_price")
            if intent.side == Side.BUY and quote.ask <= intent.limit_price:
                fill_price = min(quote.ask, intent.limit_price)
            elif intent.side == Side.SELL and quote.bid >= intent.limit_price:
                fill_price = max(quote.bid, intent.limit_price)
            else:
                return SimulatedFill(
                    ExecutionReport(
                        client_order_id=intent.client_order_id,
                        status=OrderStatus.ACCEPTED,
                        filled_volume=0.0,
                        message="limit order resting",
                        event_time=quote.time,
                    ),
                    commission=0.0,
                    point_value=self.config.point_value,
                )
        else:
            return self._reject(intent, f"unsupported order kind: {intent.kind}")

        slippage = self.config.slippage_points * self.config.point_size
        if intent.side == Side.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage

        commission = intent.volume * self.config.commission_per_volume
        return SimulatedFill(
            ExecutionReport(
                client_order_id=intent.client_order_id,
                status=OrderStatus.FILLED,
                filled_volume=intent.volume,
                avg_fill_price=fill_price,
                broker_order_id=f"sim-{intent.client_order_id}",
                message="simulated fill",
                event_time=quote.time,
            ),
            commission=commission,
            point_value=self.config.point_value,
        )

    def _reject(self, intent: OrderIntent, message: str) -> SimulatedFill:
        return SimulatedFill(
            ExecutionReport(
                client_order_id=intent.client_order_id,
                status=OrderStatus.REJECTED,
                message=message,
            ),
            commission=0.0,
            point_value=self.config.point_value,
        )
