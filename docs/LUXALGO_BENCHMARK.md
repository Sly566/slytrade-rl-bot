# LuxAlgo Benchmark and Engineering Gap Review

This review uses only LuxAlgo's public documentation. It does not claim
feature parity with a proprietary product, and no external signal or
backtest is treated as evidence of profitability.

## Public capability map

| Publicly described capability | SlyTrade status | Engineering interpretation |
| --- | --- | --- |
| Market structure, liquidity, order blocks, imbalances, premium/discount | Implemented in the causal ICT/SMC feature engine | Keep confirmation, invalidation, zone lifetime, and availability timestamps explicit. |
| Volume and money-flow style analysis | Partial | MT5 tick volume is broker activity, not centralized traded volume. Document the source and avoid exchange-order-flow claims. |
| Multi-timeframe scanning and screeners | Partial | Add bounded symbol/timeframe snapshots with per-symbol sessions, freshness, timeout, and completed-bar rules before broadening scope. |
| Alerts and webhooks | Not an ingress product | If added, authenticate and normalize external alerts into expiring, idempotent `OrderIntent` objects before risk and OMS checks. |
| Historical backtesting | Implemented | Continue using broker-specific ticks, bid/ask fills, spread/slippage/commission, and chronological holdouts. |
| Robustness testing | Partial | Add parameter perturbation, trade-sequence Monte Carlo, regime segmentation, and live-versus-paper variance reports. |
| AI-generated strategies | Not used as authority | Any external logic must be translated into causal features and independently tested against MT5/Exness data. |

## Gaps that matter most

1. **Signal provenance:** every signal should retain its source timeframe,
   confirmation time, causal availability time, invalidation time, and
   confidence.
2. **Freshness:** public archive data can lag the broker. The existing
   fail-closed quote gate is correct and must remain mandatory for demo use.
3. **Robustness evidence:** a good historical result is insufficient without
   walk-forward, cost stress, multiple seeds, lockbox evaluation, and paper or
   shadow stability.
4. **External alert safety:** webhooks are untrusted inputs; validate
   signatures, timestamps, replay keys, expiry, ordering, and risk limits.
5. **Volume semantics:** preserve the distinction between broker tick volume
   and centralized exchange volume in features, reports, and documentation.

LuxAlgo's public disclaimers also state that alerts may be delayed or fail,
simulated results are hypothetical, and execution outcomes are not
guaranteed. Those limitations support SlyTrade's separation of strategy,
risk, OMS, broker execution, reconciliation, and kill-switch controls.

Sources:

- <https://docs.luxalgo.com/docs/algos/price-action-concepts/introduction>
- <https://docs.luxalgo.com/docs/algos/signals-overlays/introduction>
- <https://docs.luxalgo.com/docs/algos/oscillator-matrix/introduction>
- <https://docs.luxalgo.com/docs/algos/screeners/s-o/introduction>
- <https://docs.luxalgo.com/docs/ai-strategy-alerts/introduction>
- <https://docs.luxalgo.com/docs/ai-backtesting/introduction>
- <https://www.luxalgo.com/blog/stress-test-your-algorithmic-trading-strategy-guide-to-avoiding-overfitting/>
- <https://www.luxalgo.com/legal/disclaimer/>
