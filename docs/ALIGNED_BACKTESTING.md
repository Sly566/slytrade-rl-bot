# Phase 17 — Fast Aligned Backtesting

Large tick files are expensive to scan repeatedly for every strategy.

Phase 17 precomputes the executable tick quote for each bar at dataset alignment time.

Aligned bars now include:

```text
quote_time
quote_bid
quote_ask
quote_mid
quote_spread
quote_age_seconds
quote_is_fresh
```

This makes repeated baseline or RL evaluations much faster because strategies can run from the aligned bars file without scanning millions of ticks each time.

## Align once

```bash
python -m slytrade.cli align-dataset \
  --bars-file data/raw/mt5_bars/symbol=XAUUSDm/timeframe=M1/year=2026/month=07/day=01.parquet \
  --ticks-file data/raw/exness_ticks/symbol=XAUUSD/year=2026/month=07/period=2026-07.parquet \
  --timeframe M1 \
  --canonical-symbol XAUUSD \
  --output-dir data/processed/datasets/xauusd_m1_2026_07
```

## Backtest from aligned bars

```bash
python -m slytrade.cli run-aligned-backtest \
  --bars-file data/processed/datasets/xauusd_m1_2026_07/bars.parquet \
  --strategy ict-bias
```

## Why this matters

A one-month XAUUSD dataset can contain millions of ticks. Without precomputed decision quotes, every strategy backtest has to scan those ticks again. With aligned bars, quote lookup is already done once and stored.
