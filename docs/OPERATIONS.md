# Paper and shadow operations

The deployment gate in `slytrade.monitoring.gates` remains the approval source of
truth. Before a demo deployment, run a `SoakMonitor` in `paper` or `shadow`
stage and require `ready` with no alerts. Call `record(healthy=False, ...)` for
failed heartbeats or execution anomalies and `check_stale()` from the scheduler.

Configure `TradingGuardrails(..., kill_switch_path="state/kill-switch.json")`
to persist a kill switch across process restarts. Drawdown breaches write the
artifact atomically; operators must review the reason before calling
`clear()`. A `RollbackArtifact` can persist the last known-good model or
configuration version and be loaded during rollback.

## Runtime loop (new)

`slytrade paper` runs the supervised paper-trading loop in
`slytrade.runtime.paper_loop`. It enforces, in order:

1. fail-closed startup validation (`RuntimeSettings.fail_closed_checks`),
2. persistent kill switch + drawdown guards,
3. trading window (weekdays + UTC hours),
4. loss circuit breaker (`max_consecutive_losses`, daily loss/trade caps, cooldown),
5. risk-budgeted sizing (`slytrade.risk.sizing`),
6. SQLite OMS/ledger journaling (restart-safe),
7. Prometheus metrics + `/healthz` + `/readyz`.

Exit orders are never blocked by the breaker or kill switch.

## Observability

- Prometheus scrape target: `:9108/metrics`. Alert on `slytrade_kill_switch`,
  `slytrade_trading_paused`, `slytrade_daily_drawdown` and broker errors.
- Structured JSON logs: `logs/slytrade.jsonl` (rotating).
- Kubernetes probes use `/healthz` (liveness) and `/readyz` (readiness).