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