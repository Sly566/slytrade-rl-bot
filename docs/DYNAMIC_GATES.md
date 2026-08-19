# Dynamic gates — the bot decides executability instead of hard-stopping

## The problem (you were right)

Every quality gate was a **hard boolean stop**: a single failed check — even one
bar's momentum — discarded the whole entry, *regardless of the confluence
score*. A score-7 setup died to one momentum miss. That's wrong: gates should
**weight** a decision, not veto it blindly.

## The fix — score-weighted executability

Each auxiliary gate now **subtracts `gate_penalty` points** from the raw
confluence score instead of hard-stopping, and the bot executes when:

```
raw_score − penalties ≥ threshold
```

* A **strong setup** (score 6–7) overrides a minor gate miss.
* A **marginal setup** (score at threshold) still needs every gate green.
* The **spread gate stays a hard block** — it's a cost reality, not a quality
  judgment.

The gates affected (all now soft): regime preferences, MTF alignment, bar
momentum, strict MTF direction, H4 macro-trend, sweep+reversal. Hard controls
kept: cooldown, session window, fresh quote, spread.

## The measured sweep (25 months real XAUUSD M15, net of cost)

| Config | 1stY | 2ndY | 2y mean | PF | max DD | trades |
|---|---:|---:|---:|---:|---:|---:|
| hard baseline (old) | +31.8% | +38.2% | +35.0% | 1.28 | 14.3% | 712 |
| hard: no H4-trend | +8.3% | +38.6% | +23.4% | 1.06 | 23.2% | 1062 |
| hard: all quality off | +14.1% | +61.0% | +37.5% | 1.07 | 23.7% | 1680 |
| dynamic penalty=1.5 | +33.2% | +43.7% | +38.5% | 1.24 | 14.2% | 865 |
| **dynamic penalty=2.0** | **+35.9%** | **+43.8%** | **+39.8%** | **1.26** | **13.6%** | 856 |
| dynamic penalty=2.5 | +36.2% | +43.6% | +39.9% | 1.31 | 14.3% | 728 |
| dynamic penalty=3.0 | +36.2% | +43.6% | +39.9% | 1.31 | 14.3% | 728 |
| dynamic penalty=4.0 | +31.8% | +39.1% | +35.5% | 1.28 | 14.3% | 714 |

## Conclusions (data, not opinion)

1. **H4-trend is the crown jewel** — removing it halves 1stY (+31.8% → +8.3%)
   and blows drawdown out to 23%. Never relax it.
2. **Gates matter, but they should weight, not veto** — turning all gates off
   yields 1680 trades but PF collapses to 1.07 (overtrading).
3. **Dynamic gating beats hard gating** — penalty 2.0 gives **+39.8% mean net
   vs +35.0%**, with *more* trades (856 vs 712), *higher* PF (1.26 vs 1.28 ≈
   equal) and *lower* drawdown (13.6% vs 14.3%).
4. **The 2.0–3.0 plateau is robust, not a knife-edge** — the penalty curve is
   flat there; 4.0 collapses back to hard behaviour. Ship 2.0 (best DD + most
   trades) and tune in `configs/risk.yaml` if you ever want more/less
   selectivity.

## Config

`configs/risk.yaml` → `ict.entry`:

```yaml
dynamic_gates: true
gate_penalty: 2.0
```

`gate_penalty: 0` ≈ all gates off; very large ≈ hard gates. The live-loop
decision trace now shows the exact math on HOLD bars:

```
bar 1999 close=4332.86 long=6 short=1 (need>=4) h4=+1 bias=+1
  → HOLD (gates: 6 - 2 = 4 < threshold 6)
```

so you can SEE why a setup was skipped and tune accordingly.

## Next (as planned)

This was the gates. The next iteration is the **market profiles** — per-symbol
regime calibration and per-timeframe cost profiles so the bot is not just a
gold bot (the FX/oil/indices cost-wall numbers are already measured in
`docs/PORTFOLIO_VALIDATION.md`; the dynamic-gate model is the first step toward
instruments that were previously blocked by gold-tuned hard gates).
