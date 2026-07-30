# Roadmap

## Phase 1: Foundation

- [x] Clean repository
- [x] Python package structure
- [x] Basic config
- [x] Tick and bar data contracts
- [x] Guardrail tests
- [x] CI workflow

## Phase 2: Tick and Bar Data Layer

- [ ] MT5 tick collector
- [ ] MT5 bar collector
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
- [ ] Tick-driven backtest loop
- [ ] Stop-loss / take-profit simulation
- [ ] Trade ledger persistence

## Phase 5: Strategy Baselines and RL

- [x] NoTrade baseline
- [x] BuyAndHold baseline
- [x] MovingAverageCross baseline
- [x] ICTBias baseline
- [x] Backtest reporting CLI
- [ ] Baseline comparison report
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
