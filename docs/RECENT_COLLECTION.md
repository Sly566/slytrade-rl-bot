# Phase 14 — Relative Lookback Collection and Coverage Feedback

The CLI can collect data relative to the current request time.

## Recent bars

```bash
python -m slytrade.cli collect-recent-bars \
  --symbol XAUUSD \
  --timeframe M1 \
  --lookback 1y \
  --chunk-size month \
  --output-dir data/raw
```

## Recent ticks

```bash
python -m slytrade.cli collect-recent-ticks \
  --symbol XAUUSD \
  --lookback 1m \
  --chunk-size day \
  --output-dir data/raw
```

Supported lookbacks:

```text
1d, 7d, 1w, 1m, 6m, 1y, 2y
```

Month and year lookbacks use deterministic day approximations:

```text
1m = 30 days
1y = 365 days
```

## Empty chunk feedback

Collection output now includes:

```text
chunks attempted
empty chunks
```

If a 1-year tick request returns only a few files and many empty chunks, it means the MT5 terminal/broker only returned tick history for part of the requested period. This is common and must be measured instead of assumed.

Bars often have longer history availability than ticks.
