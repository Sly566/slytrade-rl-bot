# Phase 4 — Backtesting Foundation

This phase introduces a minimal but production-oriented backtesting core.

## Components

```text
BarBacktestEngine
PaperBroker
OrderManagementSystem
TradeLedger
TickExecutionSimulator
PortfolioState
PerformanceMetrics
```

Backtests now use the same safe paper execution path that future paper trading will use:

```text
Strategy -> OrderIntent -> PaperBroker -> Guardrails -> OMS -> Execution -> Portfolio -> Ledger
```

## Current execution model

- Market buy fills at ask.
- Market sell fills at bid.
- Slippage is adverse to the trader.
- Limit orders rest if the quote does not cross the limit price.
- Crossed spreads are rejected by default.

## Current portfolio model

The current model is CFD-style accounting:

- balance changes when trades realize PnL or commissions are charged,
- open positions mark to market into equity,
- long and short positions are supported,
- position reversal is supported.

## Current limitations

This is the foundation, not a final institutional simulator. Still required:

- tick-driven backtest loop,
- stop-loss / take-profit order simulation,
- partial fills,
- margin checks,
- symbol-specific point value and contract sizing,
- order book or spread regime modelling,
- trade ledger persistence,
- slippage calibration from real execution logs.

## Why this comes before RL

RL training is only useful if the simulator is trustworthy. The strategy layer must not learn fills or costs that cannot exist in live execution.
