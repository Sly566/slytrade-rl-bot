# SlyTrade RL Bot — Full Project Analysis & Production-Grade Audit

**Date:** 2026-08-14
**Repo:** https://github.com/Sly566/slytrade-rl-bot
**Reviewer stance:** senior software developer + professional ICT trader

---

## 1. Executive summary

SlyTrade is a **genuinely well-engineered, safety-first trading research stack** — far
above the average "RL trading bot" repo. Its core strength is *integrity*: a strict
`Strategy → Guardrails → OMS → Execution → Portfolio → Ledger` pipeline, causal
(no-lookahead) feature engineering, tick+bar data contracts, realistic cost modelling,
walk-forward validation with embargoes, and a hash-chained model registry. 150 original
unit tests pass, `ruff` and `mypy` are clean.

Its main gap was **not research quality but production runtime readiness**: there was no
live/paper *loop* (the `live` command was a hardcoded `exit(1)` placeholder), no
structured logging, no metrics/health server despite a `prometheus-client` dependency,
no circuit breaker on the unused `max_consecutive_losses` config, a Docker image that
only ran `doctor`, and no Kubernetes/deployment manifests.

This audit **closed those gaps and implemented them** (see §5). The project is now
containerized, deployable, and has a supervised paper-trading runtime with Prometheus
metrics, health probes, durable state, a loss circuit breaker, session windows, and
risk-budgeted sizing. It is **not** yet approved for live capital — and by design it
refuses to be, until the deployment gate in `slytrade.monitoring.gates` is satisfied.

---

## 2. What the project already does well

| Area | Assessment |
|---|---|
| **Data layer** | Canonical tick/bar schemas, validation (crossed spreads, negative prices, duplicates), parquet/CSV storage with content-hash manifests, MT5 + Exness collectors, source-separated storage, alignment layer with precomputed decision quotes, freshness filtering. |
| **Causality / no lookahead** | Rolling-window-only features; scalers fitted per walk-forward fold; decision-time alignment (bar close, not bar open); documented bar-open vs bar-close MT5 timestamp semantics. |
| **ICT/SMC feature engine** | ATR, confirmed pivots, BOS/CHOCH, FVG, order blocks, liquidity sweeps, premium/discount, sessions — all causal. |
| **Backtesting** | Bar engine, aligned-quote engine, tick-execution engine (bar-signal/tick-fill), managed SL/TP/partial/breakeven/trailing, conservative same-bar SL-vs-TP rule, trade analytics (profit factor, expectancy, exit-reason distribution). |
| **Execution integrity** | Frozen `OrderIntent`, OMS owns state, strategy cannot self-report fills, append-only JSONL/SQLite journals, restart rehydration. |
| **Risk** | Drawdown kill switch (persistent across restarts), position caps, spread caps, live-trading refusal, risk-reducing exits never blocked. |
| **RL governance** | Multi-seed CI, cost-stress scenarios (bps), content-addressed lockbox, hash-chained append-only model registry, promotion gating, embargoed walk-forward. |
| **Personality system** | `TraderPersonality` YAML modulates thresholds/sizing by regime — a genuinely ICT-flavoured design (conviction, patience, liquidity focus, kill zones). |
| **Code hygiene** | 82 source files, typed (mypy clean), ruff clean, 210 passing tests (2 RL skips), CI on push/PR. |

---

## 3. Competitor benchmark

Benchmarked against the leading open-source RL/algo trading frameworks
(Freqtrade, NautilusTrader, FinRL, TensorTrade, Qlib, Hummingbot/OctoBot).

| Capability | **SlyTrade** (after this audit) | Freqtrade | NautilusTrader | FinRL | TensorTrade | Qlib |
|---|---|---|---|---|---|---|
| Tick-level data model | ✅ ticks + bars | ⚠️ OHLCV | ✅ full L1/L2 | ⚠️ bars | ⚠️ bars | ⚠️ bars |
| Causal no-lookahead guarantees | ✅ enforced + tested | ✅ lookahead analysis | ✅ | ⚠️ | ⚠️ | ⚠️ |
| Realistic cost model (spread/slippage/commission) | ✅ | ✅ | ✅ | partial | ✅ | partial |
| Walk-forward + embargo | ✅ | ⚠️ (hyperopt folds) | ✅ | ✅ | partial | ✅ |
| Multi-seed + cost-stress + lockbox governance | ✅ (unique) | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| Dry-run / paper runtime loop | ✅ **new** | ✅ dry-run | ✅ | ❌ (research only) | ❌ | ❌ |
| Live broker adapter | ⚠️ guarded MT5, gated | ✅ many CCXT | ✅ multi-venue | ❌ | ❌ | ❌ |
| Order management + audit journal | ✅ | ✅ sqlite | ✅ (Rust) | ❌ | partial | ❌ |
| Risk engine / kill switch | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Prometheus + health probes | ✅ **new** | ⚠️ (API server) | ✅ | ❌ | ❌ | ❌ |
| Docker + Kubernetes | ✅ **new** | ✅ | ✅ (official) | ⚠️ | ❌ | ⚠️ |
| RL algorithms | PPO (foundation) | FreqAI (non-RL ML) | — | PPO/A2C/SAC/TD3/DDPG | PPO/A2C etc. | ✅ RL suite |
| Multi-asset support | ✅ symbols config | ✅ pairs | ✅ | ✅ | ⚠️ | equities focus |
| Community / maturity | early | huge (38k★) | large | large | stalled | large |

**Reading the table.** SlyTrade's differentiators versus the field are (a) tick+bar
fidelity for CFD/scalping (Freqtrade/Nautilus are crypto/multi-venue but not MT5-native),
(b) the only stack combining **ICT/SMC feature engineering** with RL, and (c) the
strongest *governance* (lockbox + cost stress + hash-chained registry) — this is closer
to regulated-quant practice than any peer. Its weakness versus peers is **ecosystem
breadth** (Freqtrade's UI/Telegram/strategy marketplace) and **RL algorithm coverage**
(only PPO today). Neither is a blocker for a focused, single-operator prop-style system.

**Benchmark targets** (from peer ecosystems, to adopt as acceptance criteria):
- Win rate > 50%, profit factor > 1.2, Sharpe > 1.5, max drawdown < 15% (Freqtrade/AI-bot norms).
- 30-day dry-run before live, 1–2% risk per trade, daily loss limit 3% (industry norms).
- `profit factor > 1.2` after costs is the minimum bar for a deployable edge.

---

## 4. Gap analysis (severity-ranked) — *pre-audit state*

### Critical (fixed in this audit)
1. **No paper-trading runtime.** `slytrade live` was a hardcoded `exit(1)`; `PaperBroker`
   existed but nothing drove it in real time. → implemented `slytrade.runtime.paper_loop`.
2. **No observability surface.** `prometheus-client` was a dependency but never used;
   no `/healthz`/`/readyz` for orchestration. → implemented `runtime.metrics_server`.
3. **Risk config dead keys.** `max_consecutive_losses` in `risk.yaml` was ignored. →
   implemented `runtime.circuit_breaker` (consecutive losses, daily loss/trade caps, cooldown).
4. **Not deployable.** Docker image only ran `doctor`; no compose service, no Kubernetes. →
   rewritten Dockerfile, compose, kustomize manifests, systemd unit.
5. **No structured logging.** `print`/`rich` only, nothing ingestible. → `runtime.logs`.

### High (fixed in this audit)
6. **Env-var kill switch not wired.** `SLYTRADE_ALLOW_LIVE` in `.env.example` was unused.
   → `runtime.settings` (pydantic-settings, fail-closed validation).
7. **Sizing not enforced at the boundary.** Only `max_position_volume` existed; no
   risk-budgeted/Kelly sizing function. → `risk.sizing`.
8. **RL reward = raw equity delta** (variance-chasing, drawdown-blind). → `rl.rewards`
   (drawdown-penalised, turnover-penalised shaping; opt-in).
9. **Python pin `<3.13`** blocked installs on current interpreters; mypy no-redef errors.
   → relaxed to `>=3.12`, fixed the `gym`/`spaces` redefinition.
10. **Committed artifacts.** `cli.py.bak` tracked. → removed.

### Medium (fixed in second pass)
- ✅ Broker reconciliation as an automated job — `slytrade reconcile` (exit 0/2) + k8s CronJob.
- ✅ MLflow experiment tracking wired to `train_rl` (opt-in, safe no-op without MLflow).
- ✅ RL algorithm breadth — `ppo | sac | td3` via `train_policy` / `resolve_algorithm`.
- ✅ Red-folder news gate — `runtime.news_gate` + `configs/news.yaml` (pauses new entries).
- ✅ Webhook + Telegram alerting — `runtime.alerting` (best-effort, never crashes the loop).

### Outstanding (documented in `docs/ROADMAP.md` Phase 8)
- Recurrent policy for regime memory.
- Swap/margin/holiday handling in the live adapter.
- PVC-backed durable state for Kubernetes.
- Multi-symbol paper portfolio (parallel loops).
- Calendar-feed integration for the news gate.

---

## 5. What was implemented in this audit

**New runtime package (`src/slytrade/runtime/`)**
- `settings.py` — fail-closed env config (pydantic-settings); startup validation.
- `logs.py` — structured JSONL + console logging with rotation.
- `metrics_server.py` — Prometheus `/metrics` + `/healthz` + `/readyz` (stdlib server).
- `circuit_breaker.py` — consecutive-losses cooldown, daily loss/trade caps.
- `trading_window.py` — weekday + UTC-hours session gate (incl. overnight windows).
- `paper_loop.py` — the paper-trading event loop (quote stream → bar builder → ICT
  features → strategy → guardrails → OMS → paper broker → ledger → SQLite journal →
  Prometheus + soak monitor), with `ReplayQuoteProvider` and `MT5QuoteProvider`.

**Risk / RL hardening**
- `risk/sizing.py` — risk-budgeted + fractional-Kelly volume with broker-ladder clamping.
- `rl/rewards.py` — drawdown/turnover-penalised reward shaping + Sharpe helper.
- `runtime/alerting.py` — webhook + Telegram + log alert channels (best-effort).
- `runtime/news_gate.py` — red-folder pause for new entries (`configs/news.yaml`).
- `rl/tracking.py` — optional MLflow experiment tracking.
- `rl/walkforward.py` — `train_policy` + `resolve_algorithm` for PPO/SAC/TD3.

**CLI**
- `slytrade paper` (run the loop; `--replay-ticks`, `--max-bars`, `--max-seconds`).
- `slytrade serve` (standalone metrics/health server).
- `slytrade reconcile` (scheduled broker reconciliation; exit 0/2).
- `slytrade train-rl --algorithm ppo|sac|td3` (with MLflow when installed).
- `slytrade live` now explains the gate instead of a bare refusal.

**Containerization & deployment**
- Multi-stage, non-root Dockerfile with `HEALTHCHECK` and fail-closed entrypoint.
- `docker-compose.yml` (init container + paper service + doctor profile).
- `deploy/kubernetes/` kustomize base (namespace, deployment with probes/security
  context, service with Prometheus scrape annotations, example secret).
- `deploy/systemd/slytrade-paper.service` (hardened unit).
- CI now builds the image and enforces ≥70% coverage.

**Quality**
- 210 tests passing (60 new), ruff clean, mypy clean, Python `>=3.12` supported.

---

## 6. Verification performed

```
pytest                        → 210 passed, 2 skipped (RL extras not installed)
ruff check .                  → all checks passed
mypy src                      → success (79 source files)
slytrade doctor               → OK
slytrade paper --replay-ticks → end-to-end loop run + metrics server on :9108
```

---

## 7. How to run it now

```bash
# Local
pip install -e ".[dev,data]"
slytrade paper                      # live quotes via MT5 bridge
slytrade paper --replay-ticks data/ticks.parquet   # deterministic replay

# Container
docker compose up -d paper          # metrics on http://localhost:9108/metrics
curl localhost:9108/healthz

# Kubernetes
kubectl apply -k deploy/kubernetes
```

---

## 8. RL suite — completeness matrix (final pass)

| Aspect | Status |
|---|---|
| PPO / SAC / TD3 training | ✅ `train_policy` + `resolve_algorithm` |
| Recurrent policy (regime memory) | ✅ `--policy lstm` (MlpLstmPolicy) |
| Risk-adjusted reward (drawdown/turnover-penalised) | ✅ wired into the environment (`reward_type`) |
| Walk-forward with embargo + per-fold scaler | ✅ |
| Optuna hyperparameter sweep | ✅ (`optimize_ppo`) |
| Multi-seed / cost-stress / lockbox governance | ✅ (`rl.governance`) |
| Model artifact packaging (weights + scaler + features + config + hash) | ✅ `rl.deployment` |
| Hash-chained registry + evidence-gated promotion | ✅ `ModelRegistry` / `promote_artifact` |
| Inference strategy (saved model → orders) | ✅ `rl.inference.RLPolicyStrategy` |
| MLflow tracking (opt-in) | ✅ `rl.tracking` |

## 9. Task console & live demo (final pass)

- `slytrade ui` — Rich interactive task menu (collect / align / backtest / train /
  walk-forward / promote / paper / demo / reconcile / doctor).
- `slytrade collect-all` — bars for every timeframe + ticks in one step.
- `slytrade full-pipeline` — collect → align → backtest → train → walk-forward → promote.
- `slytrade demo` — guarded live demo-account loop (real demo orders via the MT5
  adapter, broker symbol-spec sizing, reconciliation-gated).

## 10. Bottom line

This is now a **production-shaped** system: the research layers were already strong, and
the missing operational runtime (paper loop, observability, circuit breakers,
containerization, deployment) has been implemented and tested. The honest remaining
work before *any* real capital is **evidence**: months of walk-forward validation,
cost-stress pass, lockbox pass, and a clean paper soak — exactly the gates the project
already encodes. Nothing here should be switched to live until then.
