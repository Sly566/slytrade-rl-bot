# Phase 3 — Causal ICT / Smart Money Concepts Feature Engine

The feature engine converts validated bar data into causal ICT/SMC features.

## Causality rule

For feature row `t`, the engine may only use information available at or before `t`.

This is especially important for pivots. A swing high/low centered at bar `c` with lookback `L` is not known until bar `c + L`. The engine therefore emits the pivot confirmation feature at `c + L`, not at `c`.

## Current feature groups

- ATR and normalized ATR
- volume ratio
- confirmed swing pivots
- BOS direction
- CHOCH direction
- bullish/bearish FVG flags
- FVG distance features
- bullish/bearish order block flags
- order block distance and strength features
- equal high / equal low flags
- liquidity sweep flag
- premium / discount location
- price percentile in recent range
- EMA trend context
- UTC session one-hot features

## Current implementation

```text
src/slytrade/features/ict.py
src/slytrade/features/sessions.py
```

## Test coverage

The tests include prefix-invariance checks. If the first `N` feature rows change when future bars are appended, the engine has lookahead leakage and must be rejected.
