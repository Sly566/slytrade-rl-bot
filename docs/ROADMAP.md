# Roadmap

## Phase 1: Foundation

- [x] Clean repository
- [x] Python package structure
- [x] Basic config
- [x] Tick and bar data contracts
- [x] Guardrail tests
- [x] CI workflow

## Phase 2: Tick and Bar Data Layer

- [x] Synthetic sample bar generator
- [x] Synthetic sample tick generator
- [x] MT5 tick collector
- [x] MT5 bar collector
- [x] Relative lookback bar collection
- [x] Relative lookback tick collection
- [x] Empty chunk coverage feedback
- [x] Exness archive tick downloader
- [x] Source-separated MT5 vs Exness tick storage
- [x] Dataset alignment layer
- [x] Source manifest for aligned bars/ticks
- [x] Symbol alias normalization for research datasets
- [x] Precomputed decision quotes in aligned bars
- [x] Fast aligned backtest path
- [x] Per-bar tick microstructure features
- [x] Causal ICT features embedded in aligned datasets
- [x] Dataset quality status / issues in manifest
- [x] Market data scope documented (historical L1 vs L2)
- [x] Tick schema validation
- [x] Bar schema validation
- [x] Parquet storage
- [x] Fresh-quote filtering for aligned datasets
- [x] Broker symbol specs for realistic point value

## Phase 3: Causal ICT/SMC Features

- [x] ATR
- [x] Confirmed pivots
- [x] BOS / CHOCH
- [x] FVG
- [x] Order blocks
- [x] Liquidity sweeps
- [x] Session features
- [x] Premium/discount

## Phase 4: Backtesting

- [x] Portfolio state
- [x] Quote-based execution simulator
- [x] Spread and slippage model
- [x] Basic metrics report
- [x] Backtest uses paper broker path
- [x] Backtest captures OMS order states
- [x] Backtest captures trade ledger records
- [x] Tick-aware backtest engine
- [x] Tick quote execution CLI
- [x] Bar-close decision-time alignment
- [x] Data diagnostics CLI
- [x] MT5 bridge info / symbol resolver commands
- [x] Fresh tick quote coverage diagnostics
- [x] Stale quote protection in tick backtests
- [x] Stop-loss / take-profit simulation
- [x] ATR-based managed trade exits
- [x] Partial take-profit support
- [x] Breakeven after partial TP
- [x] ATR trailing stop support
- [x] Conservative same-bar SL/TP handling
- [x] Trade analytics / exit reason metrics
- [x] Risk-reducing exits allowed after kill switch
- [ ] Trade ledger persistence
  - SQLite event journaling and OMS/ledger rehydration are implemented; broker reconciliation remains required.

## Phase 5: Strategy Baselines and RL

- [x] NoTrade baseline
- [x] BuyAndHold baseline
- [x] MovingAverageCross baseline
- [x] ICTBias baseline
- [x] ICTConfluence tuned baseline
- [x] Backtest reporting CLI
- [x] Baseline comparison report
- [x] Gymnasium environment foundation with transaction costs
- [x] PPO training hook
- [x] Embargoed walk-forward validation scaffolding
- [x] Multi-seed, lockbox, and cost-stress governance primitives
- [x] Hash-chained model registry and promotion checks

## Phase 5.5: OMS / Ledger / Paper Broker

- [x] Order state tracking
- [x] Append-only JSONL journal
- [x] Trade ledger
- [x] Paper broker execution path
- [x] Guardrails -> OMS -> Execution -> Portfolio -> Ledger flow

## Phase 6: Production

- [x] Guarded MT5 broker adapter and read-only preflight
- [x] Broker reconciliation
- [x] Monitoring
- [x] Alerts / soak monitoring
- [x] Persistent kill switch and rollback artifact
- [x] Paper-trading runtime loop (replay + live MT5 quote providers)
- [x] Loss circuit breaker (consecutive losses, daily loss/trade caps, cooldown)
- [x] Trading-window (session) enforcement
- [x] Risk-budgeted + fractional-Kelly position sizing
- [x] Structured JSON logging
- [x] Prometheus metrics + /healthz + /readyz server
- [x] Fail-closed runtime settings (env-driven, validated)
- [x] Multi-stage non-root Docker image with healthcheck
- [x] docker-compose (init + paper + doctor)
- [x] Kubernetes manifests (kustomize) + Prometheus scrape annotations
- [x] systemd unit (hardened) for non-container hosts
- [x] CI: container build + coverage threshold

## Phase 7: Evidence & hardening

- [x] RL algorithm breadth (SAC/TD3) behind the existing governance layer
- [x] MLflow experiment tracking wired to `train_rl` (opt-in)
- [x] Automated broker reconciliation job (`slytrade reconcile` + CronJob)
- [x] Economic-calendar / news gate for red-folder avoidance
- [x] Webhook + Telegram alerting on kill-switch, soak and broker errors

## Phase 8: RL suite completion

- [x] Risk-adjusted reward shaping wired into the RL environment (`reward_type`)
- [x] Recurrent policy (MlpLstmPolicy) for regime memory (`--policy lstm`)
- [x] Model artifact packaging: weights + scaler + feature columns + config + hash
- [x] Artifact registration in the hash-chained model registry
- [x] Evidence-gated promotion helper (`promote_artifact`)
- [x] Inference strategy (RLPolicyStrategy) so a saved artifact can trade

## Phase 9: Task-based CLI / GUI and live demo

- [x] `slytrade ui` — Rich interactive task menu (no flags to memorise)
- [x] `slytrade collect-all` — bars for every timeframe + ticks in one step
- [x] `slytrade full-pipeline` — collect → align → backtest → train → walk-forward → promote
- [x] `slytrade demo` — guarded live demo-account trading loop (real demo orders)
- [x] Broker symbol-spec point-value resolution for realistic demo sizing

## Phase 10: Outstanding (not yet done)

- [ ] Swap/margin/holiday handling in the live adapter
- [ ] PVC-backed durable state for Kubernetes
- [ ] Multi-symbol paper portfolio (parallel loops)
- [ ] Calendar-feed integration for the news gate (explicit events today)

## Deployment gates

The project now exposes explicit readiness primitives in
`slytrade.monitoring.gates`. Demo trading must not be enabled until the Python 3.12 environment, tests,
lint, type checks, historical validation, cost stress, seed stability, lockbox
test, paper stability, MT5 reconciliation, rollback verification, and manual
approval checks are complete. See `docs/EVALUATION_STANDARD.md`.
The RL environment is a foundation only; it is not a deployment approval or a
profitability claim. Walk-forward evaluation and model governance remain
mandatory before any policy can influence orders.
The governance primitives are implemented in `slytrade.rl.governance`, while
paper/shadow soak and rollback controls are in `slytrade.monitoring.operations`.
These components enforce evidence collection; they do not manufacture a
profitability edge or permit unsafe online self-modification.
