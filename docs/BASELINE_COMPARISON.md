# Phase 9 — Baseline Comparison Report

This phase adds a comparison command that runs all baseline strategies on the same bars file.

## Command

```bash
slytrade compare-baselines --bars-file data/raw/sample_bars.csv
```

It runs:

```text
no-trade
buy-and-hold
ma-cross
ict-bias
```

and ranks the results by final equity.

## Optional CSV output

```bash
slytrade compare-baselines \
  --bars-file data/raw/sample_bars.csv \
  --output-csv reports/baseline_comparison.csv
```

## Why this matters

This is the first benchmark gate before RL.

A future RL model should not be promoted unless it beats these baselines after realistic spread, slippage and commission assumptions.
