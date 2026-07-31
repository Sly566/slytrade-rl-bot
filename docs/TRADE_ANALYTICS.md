# Phase 22 — Trade Analytics and Risk-Reducing Exit Safety

This phase adds trade-level analytics and fixes a critical risk-control behavior.

## Analytics added

Backtest reports now include a Trade Analytics table with:

- fills,
- entry fills,
- exit fills,
- net realized PnL,
- gross profit,
- gross loss,
- profit factor,
- win rate,
- average win,
- average loss,
- expectancy,
- commission,
- order status counts,
- exit reason counts,
- rejection reason counts.

## Risk-reducing exits

Guardrails may block new/increasing exposure after a drawdown or kill-switch event, but they must not block exits that reduce open risk.

PaperBroker now detects reducing orders and allows them through even when the kill switch is active.

This prevents a bad state where the system wants to close risk but the risk engine blocks the close order.

## Why this matters

Before strategy tuning or RL, we need to know why trades win or lose:

```text
stop_loss
take_profit
partial_take_profit
trailing_stop
max_bars
rejected orders
```

This is the first feedback layer for improving ICT logic and future RL rewards.
