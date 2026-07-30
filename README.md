# SlyTrade RL Bot

Production-grade MT5 **tick-and-bar** based ICT/SMC reinforcement-learning trading system.

## Mission

Build a safe, testable, broker-neutral trading bot for day trading and scalping using:

- MT5 historical **tick data** for execution realism
- multi-timeframe **bar data** for market structure and context
- ICT / Smart Money Concepts feature engineering
- causal no-lookahead research methodology
- realistic tick-based backtesting
- paper trading before live trading
- risk guardrails and kill switches
- reinforcement learning only after the simulator is trustworthy

## Core Principles

1. No live trading by default.
2. No hardcoded broker-specific symbol assumptions.
3. Use both tick data and bar data.
4. No lookahead bias.
5. Paper trading before live trading.
6. Every order passes through risk guardrails.
7. Every order is tracked by an Order Management System.
8. Every model must be reproducible before deployment.
9. Monitoring and audit logs are mandatory.
10. Profit claims require walk-forward validation and paper trading evidence.

## Why Ticks and Bars?

Ticks provide execution realism:

- bid/ask spread
- mid price
- tick velocity
- spread expansion/contraction
- slippage modelling
- scalping-level precision

Bars provide market structure:

- M1 execution structure
- M5 target structure
- M15 intraday bias
- H1/H4 trend context
- D1/W1 liquidity context
- ICT/SMC levels

## Target Architecture

```text
MT5 Terminal
   ↓
Tick + Bar Data Collector
   ↓
Data Validator
   ↓
Tick/Bar Data Store
   ↓
Causal ICT/SMC Feature Engine
   ↓
Backtesting Engine
   ↓
Paper Trading Engine
   ↓
Strategy / RL Policy
   ↓
Risk Guardrails
   ↓
Order Management System
   ↓
Broker Adapter
   ↓
Execution Reports
   ↓
Portfolio State
   ↓
Monitoring + Audit Logs
```

## Safe Start

```bash
python -m pip install -e ".[dev,data]"
python -m slytrade.cli doctor
pytest
ruff check .
mypy src
```

## Safety Notice

This project is for research and engineering. It must not be used with real capital until:

- tick and bar data validation passes,
- feature generation is causal,
- backtesting is realistic,
- paper trading is stable,
- risk guardrails are enforced,
- live execution is explicitly approved.
