# Phase 8 — Backtest Reporting CLI

This phase adds a command-line reporting layer for baseline backtests.

## Command

```bash
slytrade run-backtest --bars-file data/raw/sample_bars.csv --strategy no-trade
```

Supported strategies:

```text
no-trade
buy-and-hold
ma-cross
ict-bias
```

## Examples

```bash
slytrade run-backtest \
  --bars-file data/raw/mt5_bars/symbol=XAUUSD/timeframe=M1/year=2026/month=01/day=01.parquet \
  --strategy buy-and-hold \
  --volume 0.1
```

```bash
slytrade run-backtest \
  --bars-file data/raw/sample_bars.csv \
  --strategy ma-cross \
  --fast-window 5 \
  --slow-window 20
```

For `ict-bias`, the reporter will compute causal ICT features if required feature columns are not already present.

## Why this matters

Before RL, every candidate strategy must be compared against simple baselines through the same backtest/paper execution path.
