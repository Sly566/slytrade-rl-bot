# Phase 23 — ICT Confluence Strategy Baseline

This phase starts Option B: strategy tuning.

The new `ICTConfluenceStrategy` is a stricter hand-crafted ICT/SMC baseline. It is designed to reduce overtrading and give RL a stronger benchmark to beat.

## Confluence inputs

The strategy scores causal features such as:

- BOS direction,
- CHOCH direction,
- liquidity sweep,
- bullish/bearish FVG,
- bullish/bearish order block,
- premium/discount,
- trend strength,
- tick rate,
- session filter,
- fresh quote requirement.

## Strategy name

```bash
ict-confluence
```

## Example

```bash
python -m slytrade.cli run-managed-backtest \
  --bars-file data/processed/datasets/xauusd_m1_2026_07/bars.parquet \
  --strategy ict-confluence \
  --symbol-spec-file data/raw/symbol_specs/XAUUSDm.json \
  --stop-loss-atr 1.0 \
  --partial-tp \
  --partial-take-profit-atr 1.0 \
  --partial-close-fraction 0.5 \
  --breakeven \
  --take-profit-atr 2.0 \
  --trailing-stop-atr 1.5
```

## Why this matters

The previous `ict-bias` baseline was intentionally simple and still overtraded. The confluence strategy requires multiple independent confirmations before entering.
