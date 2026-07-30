# Phase 11 — Tick-Driven Backtest Foundation

This phase adds a tick-aware backtest engine.

## Purpose

Bar data is used for strategy signals and ICT/SMC context. Tick data is used for execution quotes.

```text
Bars -> Strategy -> OrderIntent
Ticks -> Bid/Ask Quote -> PaperBroker execution
```

## Command

```bash
slytrade run-tick-backtest \
  --bars-file data/samples/xauusd_m1_sample.csv \
  --ticks-file data/samples/xauusd_ticks_sample.csv \
  --strategy buy-and-hold
```

## Causality assumption

The engine processes only ticks with `time_msc <= current bar time` before evaluating/executing that bar. This keeps tick usage causal, assuming bars are timestamped at the decision time.

## Current limitations

This is still a foundation:

- no stop-loss/take-profit simulation yet,
- no pending order lifecycle beyond resting limit status,
- no intra-bar strategy decisions yet,
- no partial fill model yet,
- no calibrated slippage model yet.

Those come before RL training.
