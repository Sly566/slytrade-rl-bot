# MT5 execution safety

`MT5BrokerAdapter` is the only supported path from an `OrderIntent` to
MetaTrader 5. It provides:

- explicit connection and account checks,
- quote and symbol-spec validation,
- broker-position and pending-order reconciliation,
- idempotent client-order handling through the OMS,
- guardrail evaluation using live equity and spread,
- broker retcode to execution-report translation,
- health and execution metrics,
- trading disabled by default.

Run the read-only preflight after installing the Python 3.12 project
environment and ensuring the terminal bridge is available:

```bash
python -m slytrade.cli mt5-preflight --symbol XAUUSD
```

If the terminal already has a deliberately managed position, provide it
explicitly rather than silently adopting it:

```bash
python -m slytrade.cli mt5-preflight \
  --symbol XAUUSD \
  --expected-position XAUUSDm=-0.25
```

The Linux bridge connection timeout defaults to 15 seconds and can be adjusted
with `SLYTRADE_MT5_TIMEOUT_SECONDS`. A timeout is a failed preflight, not a
reason to bypass reconciliation or enable trading.

Preflight must report a successful reconciliation. Any existing terminal
positions not explicitly supplied as expected state, or any unknown pending
orders, block new exposure. Connectivity alone is not deployment approval.
The demo gate additionally requires tests, lint, type checking, historical
validation, paper stability, MT5 reconciliation, and manual approval.
