# Phase 13 — Quote Freshness and Tick Coverage

Scalping systems must not execute on stale quotes.

Phase 13 adds quote freshness checks to diagnostics and tick-aware backtests.

## Why this matters

A tick that is merely before a bar decision time is not always good enough. If the latest tick is too old, it may not represent a tradable market quote.

The backtester now supports:

```text
max_quote_age_seconds
allow_bar_quote_fallback
```

Default behavior for CLI tick backtests is strict:

```text
--no-bar-quote-fallback
--max-quote-age-seconds 5.0
```

This means no trade is executed unless a fresh tick quote exists at the bar's causal decision time.

## Diagnostics

```bash
python -m slytrade.cli inspect-data \
  --bars-file data/raw/mt5_bars/symbol=XAUUSDm/timeframe=M1/year=2026/month=07/day=29.parquet \
  --ticks-file data/raw/mt5_ticks/symbol=XAUUSDm/year=2026/month=07/day=29.parquet \
  --timeframe M1 \
  --max-quote-age-seconds 5
```

The coverage section includes:

```text
bars_with_fresh_tick_before_decision
bars_with_stale_tick_before_decision
max_observed_quote_age_seconds
first_stale_decision_time
```

## Strict tick backtest

```bash
python -m slytrade.cli run-tick-backtest \
  --bars-file data/raw/.../day=29.parquet \
  --ticks-file data/raw/.../day=29.parquet \
  --strategy buy-and-hold \
  --max-quote-age-seconds 5 \
  --no-bar-quote-fallback
```

Use `--allow-bar-quote-fallback` only for smoke tests or synthetic sample data.
