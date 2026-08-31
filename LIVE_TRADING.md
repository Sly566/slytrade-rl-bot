# SlyTrade v0.9.15 SCALPER LIVE — Quick Start

v0.9.15 ships **6 setup kinds** with a **hybrid ladder exit** system so the bot
rides winners with partials instead of one-shot scalping everything:

| Setup kind     | Edge                                                                 | SL anchor                          | Grade profile | Size    | Order type |
|----------------|----------------------------------------------------------------------|------------------------------------|---------------|---------|------------|
| `RETEST_OB`    | v0.9.0 champion: pullback into OB after displacement                 | Beyond OB edge + 0.05 ATR          | A+/A/B/C      | Full    | Limit      |
| `RETEST_FVG`   | Price returns to a freshly-printed fair-value gap                    | Beyond FVG edge + 0.05 ATR         | A+/A/B/C      | Full    | Limit      |
| `LIQ_SWEEP`    | Wick takes out a minor swing (stop-run) → reversal displacement      | Wick extreme − max(0.30 ATR, $0.50) | B/C (quick)   | Half    | Market     |
| `BOS_CONT`     | BOS/CHoCH + displacement + vol spike — ride the impulse              | Last opposing minor swing          | **C only**    | 0.6×    | Market     |
| `DISP_TRAP`    | Fake breakout reversal — displacement candle trapped within 5 bars   | Beyond displacement candle extreme | A/B           | 0.75×   | Market     |
| `BREAKER`      | Failed OB becomes S/R after structure break — retest the breaker zone | Beyond breaker zone edge           | A+/A/B/C      | Full    | Limit      |

### v0.9.15 hybrid ladder exits (NEW)

Every scalp now uses a 3-tier partial exit instead of the old 0.85R one-shot:

| Tier | R-multiple | % closed | After this tier              |
|------|-----------|----------|------------------------------|
| TP1  | 1.0R      | 50%      | SL moves to breakeven        |
| TP2  | 2.5R      | 25%      | Runner begins ATR trailing   |
| Runner | >2.5R   | 25%      | 0.25 ATR trail + M5 CHoCH kill |

### v0.9.15 SL clamp (NEW)

Stop-loss distance is hard-clamped to `[0.5·ATR, min(3·ATR, 12pt)]`:
- **Floor 0.5 ATR**: prevents ultra-tight stops that get hunted by M1 noise
- **Ceiling min(3 ATR, 12pt)**: prevents ultra-wide stops that blow the risk budget
- The 12-point absolute cap is XAUUSD-specific; other assets use their own point values

### v0.9.15 sizing changes

- **working_lot default 0.04**: dynamic sizing scales this by `risk_pct / risk_cap`
- **risk_cap is the ONLY hard size rail**: the old 3× grade REJECT is removed — if
  vol_min forces oversize, only `risk_cap` is checked (no more double-rejection)

### v0.9.15 one-shot re-arm

After a position closes (TP or flat), the BOS_CONT one-shot flag is re-armed so
the next structural leg can fire a fresh entry without waiting for an opposite CHoCH.

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
# Champion persona (v0.9.15 default: long-only A+/A/B hybrid ladder)
slytrade live --symbol XAUUSDm --risk-cap 0.02

# Unrestricted scalper — SEE EVERYTHING FIRE (6 setups, both sides, all grades)
slytrade live --symbol XAUUSDm --risk-cap 0.02 --all --verbose
```

Flags:
- `--risk-cap 0.02` = max 2% of equity per trade (recommended for scalps)
- `--working-lot 0.04` = base lot size for dynamic scaling (default 0.04)
- `--max-open 3` = max 3 concurrent positions (default; auto-bumped to 10 when using `--all`)
- `--usd-zar 18.5` = USD/ZAR for P&L conversion
- `--live` = actually send orders (omit for dry-run)
- `--verbose` = dump zone/trigger state every 5 cycles and show WHY every signal was accepted/rejected
- `--all` = **unrestricted scalper** — 6 setup kinds, longs AND shorts, A+/A/B/C, H1+M15+M5 OBs+FVGs, Asian+off-hours

You'll see lines like:
```
[ENTRY] ticket=123456 LONG RETEST_OB 0.04 lots @ 4588.5 grade=A kz=ny_kz SL=4585.1 TP=4593.1
[TP1] ticket=123456 closed 0.02 lots @ 4593.1 → BE SL=4588.5 remaining=0.02
[TP2] ticket=123456 closed 0.01 lots @ 4598.5 remaining=0.01
[EXIT] ticket=123456 reason=M5_CHOCH_RUNNER
```

## 3. Go live

When dry-run looks good, kill dry-run with Ctrl+C and go live:

```bash
# Champion persona (conservative, hybrid ladder)
slytrade live --symbol XAUUSDm --risk-cap 0.01 --live

# Full scalper unrestricted (6 setups, both sides, all grades)
slytrade live --symbol XAUUSDm --risk-cap 0.02 --all --verbose --live
```

All positions carry magic number **260810** so the bot only manages its own trades.

## 4. Safety rules hard-coded into the live loop

- **SL clamp** (v0.9.15) — stop distance clamped to [0.5 ATR, min(3 ATR, 12pt)]
- **Min-lot floor** (0.01 for XAUUSDm) — skip if risk_pct would produce less than min lot
- **risk_cap only hard rail** (v0.9.15) — no more 3× grade REJECT; risk_cap is the sole size gate
- **Margin check** — rejects trade if margin > 95% equity
- **Max risk cap** (`--risk-cap`) caps per-trade risk
- **Max open positions** (default 3, bumped to 10 with `--all`); orphans count against the cap
- **M15 CHoCH emergency** — closes immediately if M15 CHoCH prints against
- **M5 CHoCH runner kill** (v0.9.15) — runner portion closed on M5 CHoCH against
- **Time-stop** — closes after 240 M1 bars (4h) regardless of P&L
- **Hybrid ladder** (v0.9.15) — TP1 50% @ 1R → BE; TP2 25% @ 2.5R; runner 25% ATR trail
- **One-shot re-arm** (v0.9.15) — BOS_CONT re-armed when position goes flat
- **Limit-at-zone** (v0.9.15) — RETEST/BREAKER use limit orders; DISP_TRAP/LIQ/BOS use market
- **Champion persona filters** — longs-only, ≥2 ATR stops, A+/A/B, M5+M15 OBs, London+NY only
- **500-point deviation** budget on market orders; IOC fill with RETURN fallback
- **Graceful Ctrl+C** — finishes current cycle, disconnects cleanly
- **Restart-safe orphan adoption** — existing magic=260810 positions are adopted at warmup
- **Entry slip** — longs pay half-spread + 5pts; shorts same (matches backtest)
- **Quick scalps sized smaller**: LIQ_SWEEP at 50%, BOS_CONT at 60% of tier size
- **DISP_TRAP sized at 75%**: moderate conviction fake-breakout reversal
- **Retest window extended to 120 M1 bars (2h)** in unrestricted mode
- **LIQ_SWEEP SL buffer** — `max(0.30 ATR, $0.50)` past the sweep wick

## 5. Monitoring

The loop prints one status line per M1 bar (~60 sec):
```
[time UTC] M1 <bar_time> bid=<b> ask=<a> eq=<equity><ccy> open=<n> floating=<+/-> closed=<n> wins=<w> realized=<+/-> 
```

Hybrid ladder events print as:
```
[TP1] ticket=123 closed 0.02 lots @ 4593.1 → BE SL=4588.5 remaining=0.02
[TP2] ticket=123 closed 0.01 lots @ 4598.5 remaining=0.01
[EXIT] ticket=123 reason=M5_CHOCH_RUNNER
```

## 6. Kill switch

- Ctrl+C (open positions stay on MT5 with broker SL/TP intact; bot stops managing)
- Flatten manually via MT5 terminal one-click close.

## 7. Unrestricted mode = pre-RL battle-testing

`--all` means:
- Engine shows ALL 6 setups (RETEST_OB / RETEST_FVG / LIQ_SWEEP / BOS_CONT / DISP_TRAP / BREAKER)
- Longs AND shorts fire
- Grades A+/A/B/C all emit
- FVGs included
- Retest window 120 M1 bars (~2h)
- Asian + off-hours allowed
- Quick scalps at reduced size (50-75%)

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
- `src/slytrade/strategy/signals.py` — signal engine (6 setup kinds + SL clamp)
- `src/slytrade/strategy/config.py` — strategy config (hybrid ladder + SL clamp params)
- `src/slytrade/data/features.py` — feature pipeline (includes liq-sweep detection)
- `src/slytrade/cli.py` — typer CLI (`slytrade live`)
- `start_mt5_bridge.sh` — Wine + MT5 + RPyC bridge
