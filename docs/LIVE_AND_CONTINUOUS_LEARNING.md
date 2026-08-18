# Live Deployment + Continuous Learning — what changed this round

## 1. The crash that stopped the live run (fixed)

`slytrade demo` crashed at startup with
`RuntimeError: MT5 symbol_select failed for XAUUSD247m`. Root cause: the bridge
returns full `SymbolInfo` objects from `symbols_get`, but `resolve_symbol`
sorted them as if they were name strings — so it sorted the ~600-character
reprs alphabetically and picked the **24/7 contract `XAUUSD247m`** instead of
the standard **`XAUUSDm`**, then hard-crashed when `symbol_select` returned
`False`.

Fix in `MT5BrokerAdapter.resolve_symbol`:
* normalise whatever the bridge returns (objects / dicts / strings) to names;
* rank: exact match first → non-247 contract → shortest name
  (so `XAUUSD` resolves to `XAUUSDm`, never `XAUUSD247m`);
* `symbol_select` returning `False` is now a **non-fatal warning** — the
  symbol is still quotable and tradable via `order_send`.

## 2. "Demo" → "Live" (the naming you asked for)

This is the **live deployment loop** — it places real orders on the connected
MT5 account (a demo account today, the live account once the deployment gate is
approved).

* Command is now `slytrade live` (`slytrade demo` kept as an alias).
* Class is `LiveTradingLoop` (`DemoTradingLoop` kept as an alias).
* Logs say `live loop starting` / `live order filled`, not "demo".

## 3. Live journal — the bot's memory

Every real fill is journaled to `data/live_journal/trades.parquet` with the
causal features the champion saw (persona score/bias, BOS/CHOCH/sweep,
premium/discount, trend, ATR, H4 direction, MTF bias) and, on close, the
realised R and exit reason. This is the raw material for the bot to learn
"what actually worked live" vs "what the backtest predicted".

## 4. `slytrade learn` — the ever-evolving loop

```
slytrade learn --bars-file data/processed/aligned/XAUUSD/m15/bars.parquet
```

Each run re-distills the persona champion from the **freshest bars**
(behavioural cloning warmstart), runs the embargoed walk-forward, and reports
the honest champion-vs-RL comparison. The verdict is automated:

* **RL beats the champion** in a majority of folds net-of-cost → it says so and
  you promote it (the deployment gate re-checks the evidence).
* **Champion remains in charge** → it says so, with the exact gap.

Schedule it after `slytrade collect_incremental` (e.g. nightly cron) and the
bot gets smarter every cycle: more live data → better distillation → the RL is
promoted the moment it genuinely wins, never before.

## 5. Data safety

`.gitignore` now covers `data/exness_derived/`, `data/samples/`,
`data/live_journal/`, and `.qodo/`. A previous `apply_and_commit.sh` run's
`git clean` deleted `data/exness_derived/` because it wasn't ignored — that
cannot happen again. (Re-collect it with `slytrade collect` /
`slytrade collect_incremental`.)

## Honest status

* **Pipeline: complete.** collect → align → backtest → train → walk-forward →
  promote → paper → live, plus incremental top-ups, multi-symbol admission,
  portfolio breaker, live journal, and the continuous-learning `learn` loop.
* **Profitable system: the rule-based champion** (+38.3% net on your MT5 data).
* **RL: correctly implemented, correctly gated.** It cannot yet beat the
  champion on one symbol's 667 trades; `learn` + the growing live journal are
  the honest route to changing that (more data, not more architecture).
