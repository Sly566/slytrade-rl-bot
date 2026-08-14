# Task Console & Full Pipeline

The bot ships two ways to run it:

1. **`slytrade ui`** — an interactive Rich console (task-based "GUI"). Pick a task
   from the menu, answer a couple of prompts, and the task runs end-to-end. No
   subcommand flags to memorise.
2. **`slytrade full-pipeline`** — the entire research→deployment pipeline in one
   command, from data collection to a promoted model.

## The tasks

| Task | What it does |
|---|---|
| `collect` / `collect-all` | Gathers **bars for every timeframe** (M1, M5, M15, H1, H4, D1) **plus ticks** for the symbol in one step (MT5 live, or synthetic samples for offline smoke tests). |
| `align` | Joins bars + ticks into one canonical dataset with ICT/SMC features, decision quotes and a quality manifest. |
| `backtest` | Runs the persona-adaptive ICT strategy with managed SL/TP exits and full trade analytics. |
| `train` | Trains a policy (PPO/SAC/TD3, MLP or recurrent LSTM, risk-adjusted or raw reward) and saves a deployable artifact into the model registry. |
| `walk-forward` | Embargoed walk-forward validation with an aggregate out-of-sample summary. |
| `promote` | Promotes a registered model through a stage (paper → shadow → demo) — refused unless the evidence checks pass. |
| `paper` | Runs the supervised paper-trading loop with Prometheus metrics. |
| `demo` | Runs the guarded **live demo-account** loop (real orders on the demo account). |
| `reconcile` | Read-only broker reconciliation / preflight. |
| `doctor` | Environment health check. |

## From scratch to a deployable model

```bash
# One-shot full pipeline — bars from MT5, ticks from the Exness archive
slytrade full-pipeline --symbol XAUUSD --source hybrid

# Or step through the GUI
slytrade ui
```

The `full-pipeline` flow:

```text
collect (MT5 bars all timeframes + Exness archive ticks)
  → align (ICT features + decision quotes + manifest)
  → backtest (persona-adaptive, managed exits)
  → train (artifact + registry)
  → walk-forward (out-of-sample)
  → promote (paper stage)
```

## Data sources (designated-source collection)

SlyTrade's data model is fixed and autonomous: **bars always come from MT5,
ticks always come from the Exness archive.** Collection pulls each from its
designated source and stitches them together for the pipeline.

| Source | Bars from | Ticks from | Needs MT5? |
|---|---|---|---|
| **hybrid** (default) | MT5 terminal (all timeframes) | Exness archive | ✅ |
| auto | MT5 → Exness → samples (graceful fallback) | Exness → MT5 → samples | optional |
| mt5 | MT5 | MT5 | ✅ |
| exness | resampled from Exness ticks | Exness archive | ❌ |
| samples | synthetic | synthetic | ❌ |

```bash
slytrade collect-all --symbol XAUUSD --source hybrid --lookback 1y
```

The alignment layer resolves the broker suffix automatically (`XAUUSDm` bars +
`XAUUSD` archive ticks → canonical `XAUUSD` dataset). `--source exness` remains
as a terminal-free fallback (resamples real Exness ticks to bars). Synthetic
data has no real edge — use it only to verify the machinery.

## Trained model → trading

```python
from slytrade.rl.deployment import load_model_artifact
from slytrade.rl.inference import strategy_from_artifact

model, artifact = load_model_artifact("ppo-XAUUSD-42")
strategy = strategy_from_artifact(model, artifact)   # usable in any backtest/paper engine
```
