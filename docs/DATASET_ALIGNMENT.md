# Phase 16 — Dataset Alignment and Source Manifest

SlyTrade may use bars from one source and ticks from another source.

Example:

```text
bars: MT5 bridge symbol XAUUSDm
ticks: Exness archive symbol XAUUSD
```

These are the same market conceptually, but their source symbols differ.

Phase 16 adds a dataset alignment layer that normalizes the research symbol and writes a manifest.

## Command

```bash
python -m slytrade.cli align-dataset \
  --bars-file data/raw/mt5_bars/symbol=XAUUSDm/timeframe=M1/year=2026/month=07/day=01.parquet \
  --ticks-file data/raw/exness_ticks/symbol=XAUUSD/year=2026/month=07/period=2026-07.parquet \
  --timeframe M1 \
  --canonical-symbol XAUUSD \
  --output-dir data/processed/datasets/xauusd_m1_2026_07
```

## Output

```text
data/processed/datasets/xauusd_m1_2026_07/bars.parquet
data/processed/datasets/xauusd_m1_2026_07/ticks.parquet
data/processed/datasets/xauusd_m1_2026_07/manifest.json
```

## Manifest fields

The manifest records:

- canonical symbol,
- original bar symbol,
- original tick symbol,
- bar source,
- tick source,
- timeframe,
- bar/tick row counts,
- bar and tick time ranges,
- decision-time ranges,
- fresh tick coverage diagnostics,
- file paths.

## Why this matters

The bot must know exactly what period and source it is training/backtesting on. This prevents accidental mismatches like one year of bars with only one month of ticks, or `XAUUSDm` bars with `XAUUSD` ticks that are not mapped.
