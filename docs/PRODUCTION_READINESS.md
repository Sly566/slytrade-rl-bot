# Production Readiness Assessment

**Date:** 2026-08-15 · **Updated after the "7 gaps" integration**

## 1. Solidified and proven on real data

| Layer | Status |
|---|---|
| Streaming data collection (MT5 bars + Exness ticks, MT5-first merge) | ✅ 43M ticks, no OOM |
| Data quality (100% fresh coverage, PASS manifest, stale bars dropped) | ✅ |
| Feature stack (ML + ICT/SMC + tick microstructure + sessions + MTF htf_*/mtf_bias) | ✅ |
| Tick-execution backtest with managed SL/TP/partial/breakeven/trailing | ✅ |
| OMS → guardrails → ledger → SQLite journal → persistent kill switch | ✅ |
| RL machinery (PPO/SAC/TD3, LSTM, 3 rewards, bounded episodes, registry) | ✅ |
| Runtime (paper + guarded demo loop, Prometheus, health, alerting) | ✅ |
| Ops (Docker host-user, compose, k8s, systemd, CI) | ✅ |

## 2. The 7 gaps — now addressed

| Gap | Fix |
|---|---|
| 1. Costs not modelled | `tasks.backtest` now defaults commission + slippage from `configs/risk.yaml` `costs:`; every reported number is net of costs. |
| 2. RL has no edge / no trade management | RL environment v2: entries route through the **same managed-exit engine** (ATR SL/TP, optional trailing); sparse reward = realized PnL at real exits; adopts the **full validated feature set** (ML + ICT + tick + MTF). |
| 3. ZAR currency mismatch | `slytrade.currency.CurrencyConverter` resolves USD rates live (USDZAR/ZARUSD) with a config fallback; demo sizing converts equity to USD. |
| 4. No calendar feed | `slytrade.runtime.calendar` builds the red-folder gate from a JSON/CSV file or JSON URL (impact-filtered), wired into the paper/demo loops. |
| 5. Single symbol only | `slytrade paper-multi` runs one guarded paper loop per symbol in parallel (shared metrics registry). |
| 6. No robustness evidence | `slytrade robustness` → trade-sequence Monte Carlo (P(loss), CI, drawdown), parameter perturbation (SL/TP ATR sensitivity), regime segmentation. |
| 7. Live-adapter hardening | Demo loop: margin guard (free margin ≥ 2× required), periodic broker reconciliation every 5 min, order idempotency (already), currency-aware sizing. |

## 3. Progress visibility

Every pipeline stage now prints a bold banner; the RL dataset build reports the
feature count; training and walk-forward use stable-baselines3's live progress
bar; walk-forward prints per-fold progress; robustness prints per-check lines.

## 4. Remaining (non-blocking, for later)

- Calendar feed needs a **live provider** (the schema is provider-agnostic; you
  supply a URL/file). The bot never vendors a scraper.
- Multi-symbol **live** (demo) portfolio — paper portfolio exists; live demo
  portfolio is a small extension of the same pattern.
- Beat-the-baseline promotion gate (promote only if RL walk-forward beats
  persona-adaptive **net of costs**).
- The demo/live loop's **intra-session loss breaker** (max consecutive losses /
  daily-DD pause against the live broker) is still pending wiring — the paper
  loop has it; the demo loop currently relies on server-side SL/TP + margin +
  kill-switch file.

## 5. Deployment platform (added)

The bot is now a self-hosted platform: `slytrade dashboard` serves a
mobile-first web UI (heartbeat, position, pending limit, equity, trades, log
tail) with start/stop/restart control over the supervised loop, bearer-token
auth, `/healthz`/`/readyz`, and Docker/compose + Caddy packaging. See
`docs/DEPLOYMENT_PLATFORM.md` for running it and reaching it from a phone
(Tailscale or a domain).

## 6. Bottom line

The infrastructure and the integration are now production-grade: the RL brain
adopts the validated feature stack, the trader persona, the MTF context, and the
managed exits — nothing is thrown away and nothing unseen is invented. What
remains is the *scientific* question no code can answer: whether the resulting
policy has a real out-of-sample edge. The honest gate is the walk-forward
aggregate, net of costs, and the robustness suite now measures it properly.
