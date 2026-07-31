# Symbol Specs and Realistic PnL Scaling

Backtest PnL should use broker symbol specifications, not placeholders.

Collect specs from MT5:

```bash
python -m slytrade.cli collect-symbol-spec --symbol XAUUSD
```

This writes:

```text
data/raw/symbol_specs/XAUUSDm.json
```

Use the spec during aligned backtests:

```bash
python -m slytrade.cli run-aligned-backtest \
  --bars-file data/processed/datasets/xauusd_m1_2026_07/bars.parquet \
  --strategy ict-bias \
  --symbol-spec-file data/raw/symbol_specs/XAUUSDm.json
```

For MT5 symbols, SlyTrade computes:

```text
point_value = trade_tick_value / trade_tick_size
point_size = trade_tick_size
```

For XAUUSDm from the Exness demo account, this was observed as approximately:

```text
trade_tick_size  = 0.001
trade_tick_value = 1.6532609999999999
point_value      = 1653.2609999999997
```

This makes PnL scale much closer to the broker's actual contract specification.
