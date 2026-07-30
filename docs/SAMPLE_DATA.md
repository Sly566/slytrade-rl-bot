# Phase 10 — Sample Data Generator

The project can generate deterministic sample bars and ticks without MT5.

This is useful for:

- demos,
- CI checks,
- onboarding,
- testing backtest commands,
- proving the repository works before connecting to MT5.

## Generate sample bars

```bash
slytrade generate-sample-bars --output-file data/samples/xauusd_m1_sample.csv --periods 500
```

## Generate sample ticks

```bash
slytrade generate-sample-ticks --output-file data/samples/xauusd_ticks_sample.csv --periods 2000
```

## Run baseline comparison on generated bars

```bash
slytrade compare-baselines --bars-file data/samples/xauusd_m1_sample.csv
```

## Safety note

The generated data is synthetic and should never be used to claim strategy profitability. It exists only to test the software pipeline.
