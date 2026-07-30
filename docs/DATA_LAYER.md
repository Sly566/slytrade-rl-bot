# Phase 2 — Tick and Bar Data Layer

SlyTrade uses both ticks and bars.

## Tick data

Tick data is used for execution realism and scalping intelligence:

- bid
- ask
- spread
- mid price
- tick velocity in later phases
- slippage modelling in later phases
- realistic order fill simulation in later phases

Canonical tick columns:

```text
time, time_msc, symbol, bid, ask, last, volume, volume_real, flags, spread, mid
```

## Bar data

Bar data is used for higher-level ICT/SMC structure:

- M1 execution structure
- M5 targets
- M15 intraday bias
- H1/H4 trend context
- D1/W1 liquidity context

Canonical bar columns:

```text
time, symbol, timeframe, open, high, low, close, tick_volume, spread, real_volume
```

## Storage layout

Ticks:

```text
data/raw/mt5_ticks/symbol=XAUUSD/year=2026/month=01/day=01.parquet
```

Bars:

```text
data/raw/mt5_bars/symbol=XAUUSD/timeframe=M1/year=2026/month=01/day=01.parquet
```

## CLI commands

```bash
slytrade collect-ticks --symbol XAUUSD --start 2026-01-01 --end 2026-01-02
slytrade collect-bars --symbol XAUUSD --timeframe M1 --start 2026-01-01 --end 2026-01-02 --chunk-size day
```

These commands require an MT5 Python integration to be installed and initialized in the runtime environment.
