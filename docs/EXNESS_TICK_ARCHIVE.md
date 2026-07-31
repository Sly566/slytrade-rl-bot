# Phase 15 — Exness Tick Archive Downloader

SlyTrade supports two tick sources:

```text
mt5_ticks      = ticks returned by the local MT5 terminal / bridge
exness_ticks   = ticks downloaded from the Exness public tick archive
```

## Why add Exness archive ticks?

MT5 terminal tick history can be limited by terminal cache and broker/server behavior. Exness provides a public tick-history archive, where current-month data can be daily and older data is provided as monthly or annual files.

The Exness archive is a better source for larger historical tick research, while MT5 bridge ticks remain useful for validating what the live terminal sees.

## Command

```bash
python -m slytrade.cli collect-exness-ticks \
  --symbol XAUUSD \
  --start 2025-07-01 \
  --end 2026-07-01 \
  --output-dir data/raw
```

## Relative collection

```bash
python -m slytrade.cli collect-recent-exness-ticks \
  --symbol XAUUSD \
  --lookback 1y \
  --output-dir data/raw
```

## Storage layout

```text
data/raw/exness_ticks/symbol=XAUUSD/year=2026/month=07/period=2026-07.parquet
```

## Source separation

Do not mix archive ticks with MT5 bridge ticks. They are intentionally stored separately:

```text
data/raw/mt5_ticks/
data/raw/exness_ticks/
```

Later we can compare overlapping periods to measure differences between your terminal feed and the Exness archive feed.
