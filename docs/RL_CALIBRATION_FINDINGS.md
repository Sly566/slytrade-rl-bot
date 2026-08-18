# RL Calibration Findings — what is actually missing (measured, not theorised)

All numbers below were measured on the user's real Exness XAUUSD data
(25 months, 48 141 M15 bars, 667 champion round trips, ~+0.13R/trade net of
$3.5/lot commission + 5pt slippage). Every return is in **R units** on the
same embargoed walk-forward test windows, so everything is directly
comparable to the rule-based champion.

## TL;DR

1. **The RL suite is not broken.** The environment is faithful (real
   round-trip cost, faithful persona actions, no label leak), the imitation
   stack is top-tier (inverse-sqrt class weights, class-balanced minibatches,
   AdamW+decay, DAgger, ensembles, KL-safe fine-tune), and 359 tests + ruff +
   mypy are clean.
2. **The gap is not the architecture.** More capacity makes it worse; the
   activation is neutral; a transformer is the same direction as "more
   capacity" — i.e. the wrong one — and 10–20× slower.
3. **The gap is not a missing filter.** A learned gate over the champion's
   own trade list improves R/trade and win rate but cuts total net 33–41%
   out-of-sample — its discrimination is at chance (AUC ≈ 0.5).
4. **The gap is a sample-size / learnability limit.** The RL is asked to
   re-derive a deterministic rule from that rule's own 667 demonstrations of a
   ~0.13R/trade, 29.5%-win edge. Distillation can approach the teacher but
   cannot exceed it without new information. The champion **is** the ceiling on
   this dataset — which is exactly why the promotion gate refuses RL for live.

## 1. Architecture ablation (6-fold OOS env-R, BC+DAgger warmstart)

| Policy | OOS mean | Fold spread |
|---|---:|---|
| **persona (champion)** | **+2.35** | −5.3 … +12.6 |
| mlp [64,64] tanh (current) | −4.83 | −26.3 … +18.7 |
| mlp [256,256] tanh (more capacity) | −9.50 | −19.7 … 0.0 |
| mlp [64,64] ReLU (activation) | −4.06 | −6.7 … −0.7 |
| TabTransformer (attention over features) | — | wired + verified, but 10–20× slower; same capacity direction |

**Reading:** doubling the network width made OOS **worse** (−4.8 → −9.5).
Swapping tanh→ReLU barely moved it. A transformer is strictly more parameters
plus attention over the feature axis, i.e. more of the thing that already
hurt — on a 667-label teacher. **Architecture, kwargs and activation are not
the problem.**

## 2. Feature ablation (the real overfit driver)

| Observation | OOS mean | Fold spread |
|---|---:|---|
| all 188 features | −4.83 | ±26 R |
| top-15 (constant-excluded) | +0.49 | ±8 R |
| top-12 clean | +6.35 | one fold −22.5 |
| top-20 / top-30 clean | −5.06 / −4.12 | ±37 R |

**Reading:** 188 features on ~137 labelled entries per training window is the
overfitting mechanism. Pruning dead channels collapses variance (±26R → ±8R)
and pulls OOS to roughly break-even / near-champion — but no configuration
robustly beats the champion (the top-12 "win" carries a −22.5R fold; it is
overfitting, not edge). **Simple, low-variance policies converge toward the
champion; they do not exceed it.**

## 3. Gate / filter (RL-as-filter over the champion)

| Strategy (17 folds) | Total net R | R/trade | Win rate | Trades |
|---|---:|---:|---:|---:|
| champion (ungated) | +49.69 | +0.174 | 30.0% | 438 |
| LogReg gate (τ* on val) | +29.30 (−41%) | +0.359 | 38.7% | 294 |
| HistGB gate (τ* on val) | +33.44 (−33%) | +0.335 | 33.7% | 301 |

**Reading:** filtering raises quality (R/trade, win rate) but **destroys total
net** because the classifier's out-of-sample discrimination is at chance (val
AUC ≈ 0.5; τ* chosen on validation does not generalise). The champion's trade
list contains **no reliably selectable better subset** — the champion's own
`setup_score` threshold is already the total-net optimum.

## Root cause (the honest answer to "what are we missing")

A neural policy can only beat the champion if it finds profitable behaviour
**outside the champion's support** (trades the champion would not take). The
training set is generated *by* the champion, so no such supervision exists;
generalising beyond the teacher is therefore pure guessing, and guessing loses.
667 trades of a low-win-rate, high-variance edge is simply not enough signal
for **any** learned policy to beat the deterministic rule that produced them.

## What would actually move it (ranked)

1. **More labelled trades from more markets.** This is the binding constraint
   and it is also the stated product goal (currency-agnostic profitability).
   5 000+ trades across 10+ symbols is where a learned policy can genuinely
   exceed a hand-tuned single-symbol rule. Everything else is second-order.
2. **Give the RL a different decision surface with new information** —
   regime-adaptive sizing / portfolio allocation across symbols / kill-switch
   gating — rather than re-deriving the champion's entry timing.
3. A policy-gated champion only becomes interesting once (1) or (2) supplies a
   genuinely informative signal the champion does not already use.

## What was shipped this round (calibration, not a fake win)

- `RLDataset` now **drops near-constant columns** (variance < 1e-6) instead of
  letting the scaler's variance floor inject 28 channels of pure noise into the
  observation (measured: 18 fixed-personality columns + 2 constant timeframe
  labels + 8 degenerate D1 session/killzone broadcasts on real M15 data).
  Observation shrank 188 → 159 features; no signal lost (rare-event indicators
  such as `equal_high`/`equal_low` have std ≫ 1e-6 and are kept).
- **Walk-forward defaults to `n_seeds=3`** (CLI + task + library) so the
  champion-vs-RL promotion decision can never hinge on one random seed draw
  (single-seed PPO fold variance is ±26R).
- The promotion gate (`require_champion_beat`) continues to refuse RL for
  demo/live until it beats the champion net-of-cost — which, on this dataset,
  is the correct, honest outcome.

## Bottom line

The profitable, deployable system remains the **rule-based persona champion**
(+38.3% net on the user's own MT5 2-year run, validated against the Exness
archive). The RL "superbrain" is correctly implemented, correctly calibrated,
and correctly held back from live. To make it *win* rather than *track*, the
next investment is data breadth (more markets), not another architecture.
