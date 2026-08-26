# SlyTrade v0.9.1 SCALPER LIVE — Quick Start

v0.9.1 ships **4 setup kinds** so the bot is IN the move printing money, not
sitting on hands waiting for a retest that may never come:

| Setup kind     | Edge                                                                 | SL anchor                          | Grade profile | Size    |
|----------------|----------------------------------------------------------------------|------------------------------------|---------------|---------|
| `RETEST_OB`    | v0.9.0 champion: pullback into OB after displacement                 | Beyond OB edge + 0.05 ATR          | A+/A/B/C      | Full    |
| `RETEST_FVG`   | Price returns to a freshly-printed fair-value gap                    | Beyond FVG edge + 0.05 ATR         | A+/A/B/C      | Full    |
| `LIQ_SWEEP`    | Wick takes out a minor swing (stop-run) → reversal displacement      | Beyond the sweep wick extreme      | B/C (quick)   | Half    |
| `BOS_CONT`     | BOS/CHoCH + displacement + vol spike — ride the impulse              | Last opposing minor swing          | B/C (quick)   | 0.6×    |

All scalps use the same 0.85R one-shot TP that battle-tested PF 2.00. Champion
persona (default, no flags) trades ONLY `RETEST_OB` longs A+/A/B with the
v0.9.0 long-only bias — PF 2.00, OOS 2.57 preserved. `--all` (unrestricted)
fires ALL 4 setups long+short so we gauge what needs fixing before Layer 6 RL.

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
- **Margin check** — rejects trade if margin > 95% equity
- **Max risk cap** (`--risk-cap`) caps per-trade risk
- **Max open positions** (default 3, bumped to 10 with `--all`)
- **M15 CHoCH emergency** — closes immediately if M15 CHoCH prints against
- **Time-stop** — closes after 240 M1 bars (4h) regardless of P&L
- **Champion persona filters** (longs-only, ≥2 ATR stops, A+/A/B, M5+M15 OBs, London+NY only, Asian off) are ON by default
- **30-point slippage** budget on market orders; IOC fill
- **Graceful Ctrl+C** — finishes current cycle, disconnects cleanly
- **Restart-safe** — existing positions picked up by magic number and resume monitoring
- **Entry slip** — longs pay half-spread + 5pts; shorts same (matches backtest)
- **Quick scalps sized smaller**: LIQ_SWEEP at 50%, BOS_CONT at 60% of tier size (quick in-out)
- **Retest window extended to 120 M1 bars (2h)** in unrestricted mode so M5 displacements with slow pullbacks still fire

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
