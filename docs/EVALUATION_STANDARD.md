# Trading-RL evaluation standard

A profitable single backtest is not deployment evidence. This project follows
the stricter evaluation practices used by current financial-RL research and
deployment tooling:

- chronological train/validation/test splits with purging and embargoes;
- a final untouched lockbox period;
- multiple independent random seeds with median, dispersion, and worst-seed
  reporting;
- spread, commission, latency, slippage, rejection, and partial-fill stress
  scenarios;
- risk reporting beyond return: maximum drawdown, duration, recovery,
  Sortino, Calmar, expected shortfall, turnover, exposure, and concentration;
- regime/session/instrument slices and feature/reward ablations;
- deterministic replay, paper trading, shadow mode, rollback verification, and
  human approval before demo promotion.

Relevant public references:

- [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta) for
  chronological train/test/trading workflows and financial-RL datasets.
- [FinRL-Meta paper](https://arxiv.org/abs/2211.03107) for leakage,
  survivorship-bias, and overfitting risks.
- [Stable-Baselines3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
  for separate evaluation environments and multi-seed variance.
- [ABIDES](https://github.com/abides-sim/abides) for discrete-event,
  latency-aware market-microstructure simulation.
- [CPCV implementation notes](https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html)
  for purged and embargoed cross-validation.

These references inform acceptance criteria; none is a guarantee of
profitability. A model must pass the repository's deployment gate and remain
behind the OMS, broker reconciliation, and risk guardrails.
