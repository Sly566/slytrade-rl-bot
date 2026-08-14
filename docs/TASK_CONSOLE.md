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
# One-shot full pipeline (use --source mt5 when your MT5 bridge is running)
slytrade full-pipeline --symbol XAUUSD --source auto

# Or step through the GUI
slytrade ui
```

The `full-pipeline` flow:

```text
collect (bars all timeframes + ticks)
  → align (ICT features + decision quotes + manifest)
  → backtest (persona-adaptive, managed exits)
  → train (artifact + registry)
  → walk-forward (out-of-sample)
  → promote (paper stage)
```

## Sample mode

Without an MT5 terminal, pass `--source samples` (or choose "No" for live MT5 in
the UI). This generates deterministic synthetic bars/ticks so the whole pipeline
can be smoke-tested end-to-end on any machine. Synthetic data has no real edge —
use MT5/Exness data for any evidence-based decision.

## Trained model → trading

```python
from slytrade.rl.deployment import load_model_artifact
from slytrade.rl.inference import strategy_from_artifact

model, artifact = load_model_artifact("ppo-XAUUSD-42")
strategy = strategy_from_artifact(model, artifact)   # usable in any backtest/paper engine
```
