from __future__ import annotations

from dataclasses import dataclass, field

from slytrade.execution.models import Side


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0  # positive = long, negative = short
    avg_price: float = 0.0
    point_value: float = 1.0

    @property
    def direction(self) -> int:
        if self.quantity > 0:
            return 1
        if self.quantity < 0:
            return -1
        return 0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-12

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.is_flat:
            return 0.0
        return self.quantity * (mark_price - self.avg_price) * self.point_value


@dataclass
class Fill:
    symbol: str
    side: Side
    volume: float
    price: float
    commission: float = 0.0
    point_value: float = 1.0


@dataclass
class PortfolioState:
    initial_balance: float
    balance: float | None = None
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    total_commission: float = 0.0

    def __post_init__(self) -> None:
        if self.balance is None:
            self.balance = float(self.initial_balance)

    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill and return realized PnL from any closed quantity.

        This is CFD-style accounting: balance changes only by realized PnL and
        commissions, while open exposure is tracked by positions.
        """
        if fill.volume <= 0:
            raise ValueError("fill volume must be positive")
        if fill.price <= 0:
            raise ValueError("fill price must be positive")

        signed_qty = fill.volume if fill.side == Side.BUY else -fill.volume
        position = self.positions.get(fill.symbol)
        if position is None or position.is_flat:
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=signed_qty,
                avg_price=fill.price,
                point_value=fill.point_value,
            )
            self._charge_commission(fill.commission)
            return 0.0

        realized = 0.0
        same_direction = (position.quantity > 0 and signed_qty > 0) or (position.quantity < 0 and signed_qty < 0)
        if same_direction:
            old_abs = abs(position.quantity)
            new_abs = old_abs + abs(signed_qty)
            position.avg_price = ((position.avg_price * old_abs) + (fill.price * abs(signed_qty))) / new_abs
            position.quantity += signed_qty
            self._charge_commission(fill.commission)
            return 0.0

        closing_qty = min(abs(position.quantity), abs(signed_qty))
        existing_direction = 1 if position.quantity > 0 else -1
        realized = closing_qty * existing_direction * (fill.price - position.avg_price) * position.point_value
        self.realized_pnl += realized
        assert self.balance is not None
        self.balance += realized

        remaining_existing = abs(position.quantity) - closing_qty
        remaining_new = abs(signed_qty) - closing_qty

        if remaining_existing > 1e-12:
            position.quantity = existing_direction * remaining_existing
        elif remaining_new > 1e-12:
            new_direction = 1 if signed_qty > 0 else -1
            position.quantity = new_direction * remaining_new
            position.avg_price = fill.price
            position.point_value = fill.point_value
        else:
            position.quantity = 0.0
            position.avg_price = 0.0
            self.positions.pop(fill.symbol, None)

        self._charge_commission(fill.commission)
        return realized

    def mark_to_market(self, marks: dict[str, float]) -> float:
        assert self.balance is not None
        equity = self.balance
        for symbol, position in self.positions.items():
            mark = marks.get(symbol)
            if mark is not None:
                equity += position.unrealized_pnl(mark)
        return equity

    def exposure(self, marks: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            mark = marks.get(symbol, position.avg_price)
            total += abs(position.quantity) * mark * position.point_value
        return total

    def _charge_commission(self, commission: float) -> None:
        if commission < 0:
            raise ValueError("commission cannot be negative")
        assert self.balance is not None
        self.balance -= commission
        self.total_commission += commission
