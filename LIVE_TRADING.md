# SlyTrade v0.9.13.4 SCALPER LIVE — Quick Start

v0.9.13 ships **4 setup kinds** so the bot is IN the move printing money, not
sitting on hands waiting for a retest that may never come:

| Setup kind     | Edge                                                                 | SL anchor                          | Grade profile | Size    |
|----------------|----------------------------------------------------------------------|------------------------------------|---------------|---------|
| `RETEST_OB`    | v0.9.0 champion: pullback into OB after displacement                 | Beyond OB edge + 0.05 ATR          | A+/A/B/C      | Full    |
| `RETEST_FVG`   | Price returns to a freshly-printed fair-value gap                    | Beyond FVG edge + 0.05 ATR         | A+/A/B/C      | Full    |
| `LIQ_SWEEP`    | Wick takes out a minor swing (stop-run) → reversal displacement      | Wick extreme − max(0.30 ATR, $0.50) | B/C (quick)   | Half    |
| `BOS_CONT`     | BOS/CHoCH + displacement + vol spike — ride the impulse              | Last opposing minor swing          | B/C (quick)   | 0.6×    |

All scalps use the same 0.85R one-shot TP that battle-tested PF 2.00. Champion
persona (default, no flags) trades ONLY `RETEST_OB` longs A+/A/B with the
v0.9.0 long-only bias — PF 2.00, OOS 2.57 preserved. `--all` (unrestricted)
fires ALL 4 setups long+short so we gauge what needs fixing before Layer 6 RL.

### v0.9.13 risk + restart safety

- **Hard-REJECT oversize vol_min scalps**: if the broker `volume_min` floor
  forces actual risk above `max(risk_cap, 1.5%)` **or** `3×` the grade target,
  the trade is skipped (`[REJECT] … vol_min=… forces risk=…`). Mild 1.25–3×
  oversize still only warns (`[SIZE-WARN]`).
- **Orphan adoption at startup**: after warmup, any open `magic=260810`
  position is booked into the LiveTrade ledger (`[adopt] orphan ticket=…`)
  so M15 CHoCH emergency + time-stop keep protecting it across restarts.
  `bars_held` is seeded from wall-clock age (a 3h-old orphan still times out
  after ~1 more hour, not a fresh 4h clock).

### v0.9.13.1 sleep-crash fix

- **`_clamp_sleep` guard**: every `time.sleep` in the run loop now routes
  through `LiveTrader._clamp_sleep(...)`, which clamps to `>= 0` before
  sleeping. The bar-boundary poller raced the wall clock — if a monitor call
  crossed `end_wait` mid-iteration, `end_wait - time.time()` went negative and
  `time.sleep(negative)` raised `ValueError: sleep length must be
  non-negative`, killing the trader (the 13:14 crash). NaN/±inf/garbage also
  collapse to 0.0 so no input can take the loop down.

### v0.9.13.2 loss accounting + BOS_CONT anchor fix

- **Realized P&L from broker deal history**: when a position closes between
  polls, the bot now pulls the actual fill P&L from MT5 deal history
  (`_deal_profit`) instead of the last-poll unrealized estimate. The 16:14 SL
  in v0.9.13.1 was recorded as `-9.60ZAR` while the broker really lost
  `-37.75ZAR` — the estimate understated stopped-out losses (and would have
  poisoned RL rewards). Falls back to the old estimate only if history is
  unavailable.
- **BOS_CONT uses the nearest opposing minor swing** (trigger-TF vs M1) as its
  SL anchor. The M5 ATR-ZigZag level is forward-filled and only refreshes when
  a new pivot confirms `1.5×ATR` away, so in a trend it can sit 35+ pts behind
  price (16:16 live: `risk=39.38 atr=2.06` ≈ 19×ATR) — the old code anchored
  every BOS_CONT stop to that stale level and the 0.5–7 ATR band rejected it,
  so BOS_CONT could never fire in trending/choppy markets. Now the fresh M1
  level (the one the bar actually broke FROM) is used when it's nearer.

### v0.9.13.3 truthful REJECT logging

- **`[REJECT]` log names the actual bound** the vol_min floor violated. v0.9.13.2
  always printed `risk=… > cap=…` even when it was the **3× target** rule that
  fired — the 17:00 live `BOS_CONT/C` showed `risk=3.32% > cap=5.00%` (false:
  3.32% is under the 5% cap; the true binder was `> 3× target 0.15% = 0.45%`).
  Now the line states `> cap=X%`, `> 3× target Y%`, or `> cap=X% and > 3×
  target Y%` exactly as triggered, so the log can't mislead the operator.

### v0.9.13.4 deal-history window fix (current)

- **`_deal_profit` now selects a 7-day history window.** v0.9.13.2's
  `[now-1h, now+2min]` range was interpreted by MT5 in **server time**
  (Exness MT5Trial9 = UTC+3), so the close deal was outside the window,
  `history_deals_get` returned nothing, and the bot silently fell back to the
  last-poll estimate again — the 19:17 exit booked `-3.68ZAR` while the broker
  really lost `-28.18ZAR`. The wide window covers any server offset/DST, and a
  `[WARN]` is printed if the lookup ever fails so a fallback can't pass
  unnoticed again.

## 1. Start the MT5 bridge (terminal 1)

The bridge runs MT5 under Wine and exposes an RPyC server on `127.0.0.1:18812`. Keep this terminal open:

```bash
bash start_mt5_bridge.sh
```

Wait for `Starting mt5linux RPyC server on :18812 …`. Verify in a second terminal:

```bash
slytrade mt5-info
```

## 2. Dry-run first (terminal 2)

Dry-run prints orders without sending them. Always do this first to confirm signal quality:

```bash
# Champion persona (v0.9.1 default: long-only A+/A/B RETEST_OB — matches PF 2.00)
slytrade live --symbol XAUUSDm --risk-cap 0.02

# Unrestricted scalper — SEE EVERYTHING FIRE (liq sweeps + BOS continuation + retests both sides)
slytrade live --symbol XAUUSDm --risk-cap 0.02 --all --verbose
```

Flags:
- `--risk-cap 0.02` = max 2% of equity per trade (recommended for scalps — more signals, smaller size)
- `--max-open 3` = max 3 concurrent positions (default; auto-bumped to 10 when using `--all`)
- `--usd-zar 18.5` = USD/ZAR for P&L conversion
- `--live` = actually send orders (omit for dry-run)
- `--verbose` = dump zone/trigger state every 5 cycles and show WHY every signal was accepted/rejected
- `--all` = **unrestricted scalper** — 4 setup kinds, longs AND shorts, A+/A/B/C, H1+M15+M5 OBs+FVGs, Asian+off-hours. This is what you run to see the bot scalp during liquidity grabs and momentum bursts.

You'll see lines like:
```
[ENTRY] ticket=123456 LONG LIQ_SWEEP 0.01 lots @ 4588.5 grade=C  kz=ny_kz SL=4585.1 TP=4591.2
[ENTRY] ticket=123457 SHORT BOS_CONT  0.01 lots @ 4593.2 grade=B  kz=ny_kz SL=4596.5 TP=4590.5
[ENTRY] ticket=123458 LONG RETEST_OB  0.02 lots @ 4612.1 grade=A  M15 kz=london_kz SL=4606.8 TP=4616.6
```

Let dry-run through a full London or NY session (London opens 10h SAST, NY 15h SAST) and watch it:
1. Pull fresh M1/M5/M15/H1/H4/D1 bars from MT5 every minute
2. Recompute features (ATR, swings, OBs, FVGs, CHoCH, displacement, liquidity sweeps)
3. Causally align HTFs to the last CLOSED M1 bar (no look-ahead)
4. Fire signals through the persona state machine
5. Track SL/TP/M15-CHoCH/time-stop exits

## 3. Go live

When dry-run looks good (scalps fire during moves, SL/TP distances make sense, no spam), kill dry-run with Ctrl+C and go live:

```bash
# Champion persona (conservative, PF 2.00 baseline)
slytrade live --symbol XAUUSDm --risk-cap 0.01 --live

# Full scalper unrestricted (more trades, both sides, all 4 setups)
slytrade live --symbol XAUUSDm --risk-cap 0.02 --all --verbose --live
```

All positions carry magic number **260810** so the bot only manages its own trades and never touches manual positions.

## 4. Safety rules hard-coded into the live loop

- **Min-lot floor** (0.01 for XAUUSDm) — skip if risk_pct would produce less than min lot
- **vol_min oversize REJECT** (v0.9.13) — if min-lot forces actual risk > `max(risk_cap, 1.5%)` or > `3×` target, skip the trade (`[REJECT]`); mild 1.25–3× only `[SIZE-WARN]`s
- **Margin check** — rejects trade if margin > 95% equity
- **Max risk cap** (`--risk-cap`) caps per-trade risk
- **Max open positions** (default 3, bumped to 10 with `--all`); orphans count against the cap
- **M15 CHoCH emergency** — closes immediately if M15 CHoCH prints against (including adopted orphans)
- **Time-stop** — closes after 240 M1 bars (4h) regardless of P&L; orphan age seeds `bars_held`
- **Clamped sleeps** (v0.9.13.1) — every `time.sleep` goes through `_clamp_sleep` so a negative remaining duration (wall-clock crossing `end_wait` mid-iteration) can never raise `ValueError` and kill the loop
- **Ground-truth trade P&L** (v0.9.13.2) — broker-side closes are reconciled from MT5 deal history, not the last-poll unrealized estimate, so wins/losses and realized stats match the account
- **BOS_CONT fresh anchor** (v0.9.13.2) — SL uses the nearest opposing M1/TF minor swing, so a stale M5 pivot can't inflate a scalp stop to 19×ATR and get it auto-rejected
- **Truthful REJECT logs** (v0.9.13.3) — the `[REJECT]` line names which risk bound was violated (cap vs 3× target), so a 3×-rule reject can never be misread as a cap failure
- **Server-time-safe deal lookup** (v0.9.13.4) — `_deal_profit` selects a 7-day history window (works with Exness UTC+3 server time) and warns if it has to fall back to the last-poll estimate
- **Champion persona filters** (longs-only, ≥2 ATR stops, A+/A/B, M5+M15 OBs, London+NY only, Asian off) are ON by default
- **500-point deviation** budget on market orders; IOC fill with RETURN fallback
- **Graceful Ctrl+C** — finishes current cycle, disconnects cleanly
- **Restart-safe orphan adoption** — existing `magic=260810` positions are adopted at warmup and resume CHoCH/time-stop monitoring
- **Entry slip** — longs pay half-spread + 5pts; shorts same (matches backtest)
- **Quick scalps sized smaller**: LIQ_SWEEP at 50%, BOS_CONT at 60% of tier size (quick in-out)
- **Retest window extended to 120 M1 bars (2h)** in unrestricted mode so M5 displacements with slow pullbacks still fire
- **LIQ_SWEEP SL buffer** (v0.9.12+) — `max(0.30 ATR, $0.50)` past the sweep wick so retests don't clip the stop

## 5. Monitoring

The loop prints one status line per M1 bar (~60 sec):
```
[time UTC] M1 <bar_time> bid=<b> ask=<a> eq=<equity><ccy> open=<n> floating=<+/-> closed=<n> wins=<w> realized=<+/->
```

Diagnostics print every 5 cycles with `--verbose`:
```
[diag] M1 last 30M1: {'bear_disp': 3, 'bear_liq_sweep': 1} bull_disp=12:04 bear_disp=12:18 bull_sweep=None bear_sweep=12:17
[diag] M5 last 60M1: {'bear_disp': 4, 'minor_bos_dn': 2} bull_disp=11:55 bear_disp=12:15 bull_sweep=None bear_sweep=None
[latest M1 bar flags] bear_disp, minor_bos_dn
```

Watch `[diag]` during big moves: if `bear_disp`/`bull_disp`/`*_liq_sweep` counters aren't incrementing during $10+ legs, that's a feature bug; if they ARE incrementing but no `[ENTRY]` fires, the rejection reasons in `[SIG]` lines will tell us why.

Pipe to log:
```bash
slytrade live --symbol XAUUSDm --risk-cap 0.02 --all --verbose --live 2>&1 | tee -a live.log
```

## 6. Kill switch

- Ctrl+C (open positions stay on MT5 with broker SL/TP intact; bot stops managing)
- Flatten manually via MT5 terminal one-click close.

## 7. Unrestricted mode = pre-RL battle-testing

This is the mode you run to see the bot scalp — `--all` does NOT mean reckless.
It means:
- Engine shows ALL setups (RETEST_OB / RETEST_FVG / LIQ_SWEEP / BOS_CONT)
- Longs AND shorts fire (champion only allows longs because shorts PF 1.01)
- Grades A+/A/B/C all emit (champion blocks C)
- FVGs included (champion only trades OBs)
- Retest window 120 M1 bars (~2h) vs champion's 60 (~1h)
- Asian + off-hours allowed (we still filter dead ATR bars globally)
- Quick scalps at reduced size (50-60%)

This is how we see what's being left on the table during sell-offs, spikes,
and liquidity grabs BEFORE we hand anything to Layer 6 RL. If PF is bad in
unrestricted mode, RL won't fix it — we fix the signal engine first.

## 8. Session cheat-sheet (UTC, SAST = UTC+2)

| Session       | UTC        | SAST       | Allowed by default? |
|---------------|------------|------------|---------------------|
| Asian kz      | 00:00-03:00| 02:00-05:00| Only w/ `--all`     |
| London open30 | 08:00-08:30| 10:00-10:30| Yes                 |
| London kz     | 07:00-10:00| 09:00-12:00| Yes                 |
| NY open30     | 13:30-14:00| 15:30-16:00| Yes                 |
| NY kz         | 12:00-15:00| 14:00-17:00| Yes                 |

## Files
- `src/slytrade/live/trader.py` — main loop (LiveTrader class + argparse CLI)
- `src/slytrade/strategy/signals.py` — signal engine (4 setup kinds)
- `src/slytrade/data/features.py` — feature pipeline (includes liq-sweep detection)
- `src/slytrade/cli.py` — typer CLI (`slytrade live`)
- `start_mt5_bridge.sh` — Wine + MT5 + RPyC bridge
