# Data and Feature Pipeline Completion Gate

The data layer is considered production-grade for the current historical research scope when the following are true:

## Historical sources

- MT5 bars are collected and stored separately under `data/raw/mt5_bars/`.
- Exness archive ticks are collected and stored separately under `data/raw/exness_ticks/`.
- MT5 bridge ticks are kept separately under `data/raw/mt5_ticks/` for terminal-feed validation.

## Alignment

- Bars and ticks are aligned into `data/processed/datasets/...`.
- Symbol aliases are normalized, e.g. `XAUUSDm` + `XAUUSD` -> `XAUUSD`.
- A manifest records sources, periods, symbols, row counts and fresh quote coverage.
- Aligned bars include precomputed decision-time quotes.

## Features embedded in aligned bars

Aligned bars include:

- causal ICT/SMC features,
- session features,
- per-bar tick microstructure features,
- execution quote fields.

## Truthfulness rule

Historical Exness archive ticks are Level 1 bid/ask ticks, not historical L2 order book depth. The project must not call derived tick features "L2".

## Next layers may assume

After this gate, backtesting and RL can use aligned bars as the main research table. Full tick files remain available for validation and advanced execution modelling, but repeated baseline/RL runs should use the precomputed aligned bars path for speed.

## Final hardening additions

- Alignment can drop stale bars with `--fresh-only` (default) so downstream backtests/RL do not execute on stale quotes.
- The manifest records original source bar rows, final aligned rows, dropped stale bars and quality status.
- Broker symbol specifications can be saved from MT5 and used for realistic point size / point value in backtests.
- Full historical L2 is explicitly out of scope because Exness archive ticks provide Level 1 bid/ask data, not order-book depth.
