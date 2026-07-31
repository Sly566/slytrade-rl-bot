# Phase 21 — Advanced Managed Trade Behaviour

This phase upgrades managed backtesting beyond a single full-position SL/TP.

## Added capabilities

- partial take-profit,
- configurable partial close fraction,
- move stop-loss to breakeven after partial TP,
- optional ATR trailing stop,
- conservative same-bar handling,
- deterministic price-level exit fills.

## Example

```bash
python -m slytrade.cli run-managed-backtest \
  --bars-file data/processed/datasets/xauusd_m1_2026_07/bars.parquet \
  --strategy ict-bias \
  --symbol-spec-file data/raw/symbol_specs/XAUUSDm.json \
  --stop-loss-atr 1.0 \
  --partial-tp \
  --partial-take-profit-atr 1.0 \
  --partial-close-fraction 0.5 \
  --breakeven \
  --take-profit-atr 2.0 \
  --trailing-stop-atr 1.5
```

## Conservative assumptions

If a bar touches both stop-loss and take-profit, stop-loss wins by default because the exact tick sequence inside the bar is not known in the aligned-bar engine.

## Remaining future precision work

Exact intra-bar tick-time sequencing can be implemented later with a full tick-managed executor. The aligned-bar engine is optimized for fast repeated research and uses conservative price-level fills.
