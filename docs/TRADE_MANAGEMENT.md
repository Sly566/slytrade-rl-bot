# Phase 20 — Stop-Loss / Take-Profit Trade Management

This phase adds the first deterministic managed-trade layer.

## Current scope

- one open managed trade at a time,
- entry comes from an existing baseline strategy,
- exits are owned by the managed backtest engine,
- stop-loss and take-profit are ATR based,
- if SL and TP are both touched in the same bar, stop-loss wins by default.

## Command

```bash
python -m slytrade.cli run-managed-backtest \
  --bars-file data/processed/datasets/xauusd_m1_2026_07/bars.parquet \
  --strategy ict-bias \
  --symbol-spec-file data/raw/symbol_specs/XAUUSDm.json \
  --stop-loss-atr 1.0 \
  --take-profit-atr 2.0
```

## Why this matters

RL should not learn from an environment where entries never exit realistically. This phase creates the first explicit trade lifecycle.

## Future upgrades

- partial take-profits,
- breakeven logic,
- trailing stops,
- exact tick-time exit price modelling,
- multiple simultaneous positions if portfolio research requires it.
