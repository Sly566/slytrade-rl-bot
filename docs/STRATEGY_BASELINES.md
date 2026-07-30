# Phase 7 — Strategy Baselines

Before reinforcement learning, SlyTrade needs simple baseline strategies.

RL is only useful if it beats simple alternatives after realistic costs.

## Baselines added

```text
NoTradeStrategy
BuyAndHoldStrategy
MovingAverageCrossStrategy
ICTBiasBaselineStrategy
```

## Why baselines matter

Every future model should be compared against:

1. no trading,
2. buy and hold,
3. simple trend following,
4. simple causal ICT/SMC bias.

If an RL model does not outperform these baselines after spread, slippage and commission, it should not be promoted.

## Current strategy path

All baseline strategies emit `OrderIntent` objects.

They do not directly modify portfolio state or assume fills.

```text
Strategy -> OrderIntent -> BacktestEngine -> PaperBroker -> Guardrails -> OMS -> Execution -> Portfolio -> Ledger
```
