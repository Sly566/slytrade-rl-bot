from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from slytrade.execution.journal import JsonlJournal
from slytrade.execution.models import OrderIntent, Side


@dataclass(frozen=True)
class TradeRecord:
    client_order_id: str
    symbol: str
    side: Side
    volume: float
    price: float
    commission: float
    realized_pnl: float
    reason: str
    event_time: datetime = field(default_factory=lambda: datetime.now(UTC))


class TradeLedger:
    """Append-only in-memory trade ledger with optional JSONL persistence."""

    def __init__(self, journal: JsonlJournal | None = None):
        self.records: list[TradeRecord] = []
        self.journal = journal

    def record_fill(
        self,
        intent: OrderIntent,
        *,
        volume: float,
        price: float,
        commission: float,
        realized_pnl: float,
        event_time: datetime,
    ) -> TradeRecord:
        record = TradeRecord(
            client_order_id=intent.client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            volume=volume,
            price=price,
            commission=commission,
            realized_pnl=realized_pnl,
            reason=intent.reason,
            event_time=event_time,
        )
        self.records.append(record)
        if self.journal is not None:
            self.journal.append("trade_record", {"trade": record})
        return record

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])

    @property
    def total_realized_pnl(self) -> float:
        return sum(record.realized_pnl for record in self.records)

    @property
    def total_commission(self) -> float:
        return sum(record.commission for record in self.records)
