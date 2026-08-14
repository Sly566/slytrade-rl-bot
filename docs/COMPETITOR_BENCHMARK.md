# Competitor Benchmark — RL & Algo Trading Frameworks

How SlyTrade compares to the leading open-source trading/RL frameworks, and what each
competitor does that we should (or should not) adopt.

## 1. Field overview

| Framework | Focus | RL | Live exec | Container/Deploy | Maturity |
|---|---|---|---|---|---|
| **Freqtrade** | Crypto bots | FreqAI (adaptive ML, non-RL) | ✅ dry-run + live | ✅ docker/compose, UI, Telegram | Very high (38k★) |
| **NautilusTrader** | Institutional multi-asset | — | ✅ multi-venue | ✅ official docker | High (Rust core) |
| **FinRL** | RL research (stocks/crypto) | ✅ PPO/A2C/SAC/TD3/DDPG | ❌ research only | ⚠️ | High (15k★) |
| **TensorTrade** | RL framework | ✅ PPO/A2C | ❌ simulation-first | ❌ | Stalled |
| **Qlib (Microsoft)** | ML/RL research platform | ✅ RL suite | ❌ research | ⚠️ | High |
| **Hummingbot / OctoBot** | Market-making bots | ❌ | ✅ | ✅ | High |
| **SlyTrade** | MT5 CFD scalping/day-trading, ICT/SMC + RL | PPO (foundation) | ⚠️ guarded MT5 + paper | ✅ **now** | Early |

## 2. Head-to-head by capability

### 2.1 Data realism
SlyTrade is **ahead of every RL peer** on execution realism for CFD/scalping: canonical
tick + bar schemas, bid/ask spread, freshness filtering, decision-time alignment, and a
tick-execution backtest engine (bar signals, tick fills). Freqtrade/Nautilus are
crypto/multi-venue; FinRL/TensorTrade/Qlib are bar-level and research-oriented.

### 2.2 Bias control (the "does it lie to itself?" test)
SlyTrade: causal-only features, per-fold scalers, embargoed walk-forward, lockbox,
cost-stress, hash-chained registry. **No competitor combines all of these.** Freqtrade
has lookahead-analysis tooling; NautilusTrader has deterministic simulation parity;
FinRL/TensorTrade leave bias control largely to the user.

### 2.3 Risk & operations
SlyTrade: kill switch, drawdown guards, position/spread caps, loss circuit breaker,
session window, risk-budgeted sizing. On par with Freqtrade/Nautilus; far beyond
FinRL/TensorTrade/Qlib (which have no execution risk layer).

### 2.4 RL capability
SlyTrade currently lags **FinRL** (many algorithms) and **Qlib** (nested RL). Only PPO
is wired today. Recommendation: add SAC/TD3 wrappers behind the existing governance,
plus a recurrent policy for regime memory — the governance layer is already algorithm-
agnostic (`slytrade.rl.governance`).

### 2.5 Ecosystem & operator tooling
Freqtrade wins decisively (FreqUI, Telegram control, strategy marketplace, plotly).
Adopting a Telegram/webhook alerter for kill-switch/soak events is the single
highest-leverage ecosystem feature to add next.

## 3. What to copy, what to ignore

| Adopt from peer | Rationale |
|---|---|
| Freqtrade dry-run discipline + Telegram alerts | Operator feedback loop |
| NautilusTrader risk-engine/execution separation | Already partially present; keep sharpening |
| FinRL/Qlib algorithm breadth | SAC/TD3 + recurrent policies |
| Freqtrade `--edge` / profit-factor acceptance bar | Gate promotion on profit factor > 1.2 |

| Reject from peer | Rationale |
|---|---|
| FinRL "research-only" stance | SlyTrade targets production |
| TensorTrade's stalled maintenance | Avoid as a dependency |
| Rule-only mechanical signals | SlyTrade's ICT + personality layer is its edge |

## 4. SlyTrade's defensible edge

1. **ICT/SMC feature engineering** — the only RL stack that natively reasons about
   order blocks, FVGs, liquidity sweeps, BOS/CHOCH, premium/discount, and kill zones.
2. **Personality-conditioned policy** — a single policy modulated by a trader persona +
   regime mode vector (no peer has this).
3. **Governance depth** — lockbox + cost stress + hash-chained registry ≈ regulated-quant
   practice, unique among open-source RL trading bots.
4. **MT5-native tick+bar fidelity** — the exact data model required for gold/forex
   scalping, which crypto/equity frameworks do not provide.
