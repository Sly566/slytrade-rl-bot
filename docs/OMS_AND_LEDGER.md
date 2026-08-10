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
JsonlJournal / SqliteJournal
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

## Durable journals

`JsonlJournal` remains useful for simple audit exports. `SqliteJournal` is the
durable runtime option: it stores ordered, transactional events and allows the
OMS and trade ledger to rehydrate after a process restart.

Persistence is necessary but not sufficient for live trading. A broker adapter
must still reconcile open orders and positions before it is allowed to submit
new exposure.

## Paper Broker

`PaperBroker` is the first safe broker implementation. It routes all orders through:

1. risk guardrails,
2. OMS,
3. execution simulator,
4. portfolio accounting,
5. trade ledger.

This is the path all future backtest, paper, and live execution adapters must respect.
