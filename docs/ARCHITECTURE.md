# Architecture

SlyTrade RL Bot is designed as a layered trading system.

## Data Intelligence

The bot uses both:

1. Tick data
2. Bar data

Ticks provide execution realism. Bars provide structure.

## Layers

1. Data ingestion
2. Data validation
3. Tick/bar storage
4. Feature engineering
5. Backtesting
6. Strategy / RL policy
7. Risk guardrails
8. Order management
9. Broker adapters
10. Monitoring and audit logs

## Hard Rule

The strategy layer may suggest trades, but it may not directly execute trades.

All trades must pass through:

```text
OrderIntent -> RiskGuardrails -> OMS -> BrokerAdapter -> ExecutionReport
```
