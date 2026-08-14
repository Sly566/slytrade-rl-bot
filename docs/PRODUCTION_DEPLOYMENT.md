# Production Deployment Guide

How to run SlyTrade as a supervised, containerized, observable service. All paths
below are **paper** by default and refuse to trade live until the deployment gate is
satisfied and `SLYTRADE_ALLOW_LIVE=1` + `SLYTRADE_STAGE=demo` are explicitly set.

## 1. Runtime components

| Component | Module | Purpose |
|---|---|---|
| Paper loop | `slytrade.runtime.paper_loop` | Streams quotes → ICT features → strategy → guardrails → OMS → paper broker → ledger |
| Circuit breaker | `slytrade.runtime.circuit_breaker` | Consecutive-loss cooldown, daily loss/trade caps (from `configs/risk.yaml`) |
| Trading window | `slytrade.runtime.trading_window` | Weekday + UTC-hours gate |
| News gate | `slytrade.runtime.news_gate` | Red-folder (news) pause for new entries (`configs/news.yaml`) |
| Alerting | `slytrade.runtime.alerting` | Webhook + Telegram + log channels (best-effort, never crash the loop) |
| Metrics server | `slytrade.runtime.metrics_server` | `/metrics`, `/healthz`, `/readyz` |
| Sizing | `slytrade.risk.sizing` | Risk-budgeted + fractional-Kelly volume |
| Settings | `slytrade.runtime.settings` | Fail-closed env config (pydantic-settings) |
| Tracking | `slytrade.rl.tracking` | Optional MLflow experiment tracking for `train-rl` |

## 2. Environment variables

See `.env.example`. The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `SLYTRADE_ALLOW_LIVE` | `0` | Must be `1` **and** stage `demo` to even consider live |
| `SLYTRADE_STAGE` | `paper` | `dry_run` / `paper` / `shadow` / `demo` |
| `SLYTRADE_SYMBOL` | `XAUUSD` | Traded symbol |
| `SLYTRADE_TIMEFRAME` | `M1` | Signal/decision timeframe |
| `SLYTRADE_METRICS_PORT` | `9108` | Prometheus + health port |
| `SLYTRADE_METRICS_BIND` | `0.0.0.0` | Bind address |
| `SLYTRADE_TRADING_DAYS` | `mon..fri` | Allowed weekdays |
| `SLYTRADE_TRADING_START_UTC` / `_END_UTC` | `00:00`/`23:59` | Trading hours (UTC) |
| `SLYTRADE_INITIAL_BALANCE` | `100000` | Paper account equity |
| `SLYTRADE_ALERT_WEBHOOK_URL` | `` | Generic JSON webhook for alerts (Slack/Teams/Discord) |
| `SLYTRADE_ALERT_TELEGRAM_BOT_TOKEN` / `_CHAT_ID` | `` | Telegram alerting (both required to enable) |
| `SLYTRADE_NEWS_ENABLED` | `0` | Enable the red-folder news gate |
| `SLYTRADE_NEWS_CONFIG_FILE` | `configs/news.yaml` | News event windows |

## 3. Local

```bash
python -m pip install -e ".[dev,data]"
slytrade doctor
slytrade paper --replay-ticks data/ticks.parquet --max-bars 500   # replay
slytrade paper                                                     # live via MT5 bridge
slytrade serve                                                     # metrics only
```

## 4. Docker

```bash
docker compose up -d paper          # paper loop + metrics on :9108
curl -fs localhost:9108/healthz
docker compose logs -f paper
docker compose --profile doctor up doctor   # one-off environment check
```

The image is multi-stage and non-root; the entrypoint (`docker/entrypoint.sh`) refuses
to boot if `SLYTRADE_ALLOW_LIVE=1` without `SLYTRADE_STAGE=demo`.

## 5. Kubernetes

```bash
kubectl apply -k deploy/kubernetes
kubectl -n slytrade get pods
kubectl -n slytrade port-forward svc/slytrade-paper 9108:9108
curl localhost:9108/healthz && curl localhost:9108/readyz && curl localhost:9108/metrics
```

The pod template carries Prometheus scrape annotations (`prometheus.io/scrape: "true"`),
liveness/readiness probes against `/healthz`/`/readyz`, `runAsNonRoot`, a read-only
root filesystem, dropped capabilities, and `emptyDir` volumes for state/logs (use a
PVC for durable state in production).

A `CronJob` (`slytrade-reconcile`) runs `slytrade reconcile` every 30 minutes; it
exits non-zero when broker state diverges from expectations. It requires network
access to the MT5 bridge.

## 6. Observability

- **Prometheus** — scrape `/metrics` on port 9108. Key series: `slytrade_equity`,
  `slytrade_daily_drawdown`, `slytrade_total_drawdown`, `slytrade_kill_switch`,
  `slytrade_trading_paused`, `slytrade_orders_total{status}`, `slytrade_trades_total{outcome}`.
- **Logs** — JSON lines in `logs/slytrade.jsonl` (rotating), ideal for Loki/Fluentd.
- **Alerts to add** — `slytrade_kill_switch == 1`, `slytrade_trading_paused == 1`,
  `slytrade_news_paused == 1`, `slytrade_daily_drawdown > 0.03`,
  `rate(slytrade_broker_errors_total[5m]) > 0`.
- **Operator alerts** — configure `SLYTRADE_ALERT_WEBHOOK_URL` (generic JSON) or the
  Telegram bot token/chat id; the loop emits kill-switch (critical), broker-error
  (warning), soak (warning) and shutdown-summary (info) alerts. Transports are
  best-effort and can never crash the loop.

## 7. State & rollback

- OMS + ledger events are journaled to `state/execution-events.db` (SQLite) and
  rehydrated on restart — orders and trades survive crashes.
- The kill switch persists to `state/kill-switch.json`; review the reason, then call
  `clear_kill_switch()` (or delete the file) only after operator review.
- `RollbackArtifact` (in `slytrade.monitoring.operations`) pins the last known-good
  model/config version for rollback.

## 8. Go-live sequence (non-negotiable)

1. `SLYTRADE_STAGE=paper` soak — no alerts, stable equity, ≥ N days.
2. `SLYTRADE_STAGE=shadow` — run against live quotes without orders.
3. Complete every check in `slytrade.monitoring.gates.DEFAULT_DEMO_GATE`
   (tests, lint, type check, historical validation, cost stress, seed stability,
   lockbox, paper stability, MT5 reconciliation, rollback verification, manual approval).
4. Only then consider `SLYTRADE_STAGE=demo` with real (demo) broker account and
   `SLYTRADE_ALLOW_LIVE=1`.
