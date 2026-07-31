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
- [ ] Tick schema validation
- [ ] Bar schema validation
- [ ] Parquet storage

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
- [ ] Stop-loss / take-profit simulation
- [ ] Trade ledger persistence

## Phase 5: Strategy Baselines and RL

- [x] NoTrade baseline
- [x] BuyAndHold baseline
- [x] MovingAverageCross baseline
- [x] ICTBias baseline
- [x] Backtest reporting CLI
- [x] Baseline comparison report
- [ ] Gymnasium environment
- [ ] PPO baseline
- [ ] Walk-forward validation

## Phase 5.5: OMS / Ledger / Paper Broker

- [x] Order state tracking
- [x] Append-only JSONL journal
- [x] Trade ledger
- [x] Paper broker execution path
- [x] Guardrails -> OMS -> Execution -> Portfolio -> Ledger flow

## Phase 6: Production

- [ ] MT5 broker adapter
- [ ] Broker reconciliation
- [ ] Monitoring
- [ ] Alerts
- [ ] Kill switch
