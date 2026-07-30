# Phase 5 — OMS, Trade Ledger and Paper Broker

This phase introduces the first production-style execution path.

## Execution path

```text
OrderIntent
   ↓
TradingGuardrails
   ↓
OrderManagementSystem
   ↓
TickExecutionSimulator
   ↓
PortfolioState
   ↓
TradeLedger
   ↓
JsonlJournal
```

## Order Management System

The OMS owns order state. Strategy and RL code must not assume fills.

The OMS tracks:

- client order ID
- status
- broker/simulator order ID
- filled volume
- average fill price
- last message
- timestamps

## Trade Ledger

The trade ledger stores realized fill records:

- client order ID
- symbol
- side
- volume
- fill price
- commission
- realized PnL
- reason
- timestamp

## JSONL journal

The JSONL journal is an append-only audit trail. It is intentionally simple and can later be backed by SQLite/Postgres, but even at this stage every order and trade event is inspectable.

## Paper Broker

`PaperBroker` is the first safe broker implementation. It routes all orders through:

1. risk guardrails,
2. OMS,
3. execution simulator,
4. portfolio accounting,
5. trade ledger.

This is the path all future backtest, paper, and live execution adapters must respect.
