# SlyTrade v0.9.0 LIVE — Quick Start

Champion persona PF 2.00 (longs-only, 0.85R one-shot, ≥2 ATR stops, grades A+/A/B, M5+M15 OBs, London/NY).

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
slytrade live --symbol XAUUSDm --risk-cap 0.01
```

Flags:
- `--risk-cap 0.01` = max 1% of equity per trade (safe default for your ZAR 2039 demo)
- `--max-open 3` = max 3 concurrent positions (default)
- `--usd-zar 18.5` = USD/ZAR for P&L conversion (default 18.5 — update if rate moves)
- Default is dry-run; omit `--live` to just print what it WOULD do.

You'll see lines like:
```
[2026-08-26 10:05:00 UTC] M1 10:04  bid=3452.500 ask=3452.700  eq=2039.70ZAR  open=0  floating=+0.00  closed=0 wins=0 realized=+0.00
    [DRY-RUN] BUY 0.01 XAUUSDm @ 3452.80 SL=3448.20 TP=3456.71  (L5 A M5 london_kz)
```

Let dry-run for 30-60 minutes through a session (London open 10h SAST or NY 15h SAST) and watch it:
1. Pull fresh M1/M5/M15/H1/H4/D1 bars from MT5 every minute
2. Recompute features (ATR, swings, OBs, FVGs, CHoCH, displacement)
3. Causally align HTFs to the last CLOSED M1 bar (no look-ahead)
4. Fire signals through the v0.9.0 champion persona state machine
5. Track SL/TP/M15-CHoCH/time-stop exits

## 3. Go live

When dry-run looks good (signals fire during London/NY, SL/TP are sensible distances, no spam), kill dry-run with Ctrl+C and go live:

```bash
slytrade live --symbol XAUUSDm --risk-cap 0.01 --live
```

All positions carry magic number **260810** so the bot only manages its own trades and never touches manual positions.

## 4. Safety rules hard-coded into the live loop

- **Min-lot floor respected** (0.01 for XAUUSDm) — if risk_pct would produce less than min lot, skip
- **Margin check** — rejects trade if margin > 95% equity
- **Max risk cap** (`--risk-cap`) overrides tiered sizing to keep you at or below your chosen per-trade risk
- **Max open positions** (default 3) — hedging mode but capped
- **M15 CHoCH emergency** — closes position immediately if M15 prints CHoCH against direction
- **Time-stop** — closes after 240 M1 bars (4h) if price is underwater
- **Champion persona filters** (longs-only, ≥2 ATR stops, A+/A/B, M5+M15 OBs, London+NY only, Asian off) are ON by default
- **30-point slippage** budget on market orders; IOC fill
- **Graceful Ctrl+C** — finishes current cycle then disconnects cleanly
- **Restart-safe** — if you restart the bot it picks up existing positions by magic number and resumes monitoring
- **Entry slip modelled** — longs pay half-spread + 5pts slip on fill; shorts pay half-spread + 5pts slip (matches backtest)

## 5. Monitoring

The loop prints one status line per M1 bar (~every 60 seconds):
```
[time UTC] M1 <bar_time> bid=<b> ask=<a> eq=<equity><ccy> open=<n> floating=<+/-> closed=<n> wins=<w> realized=<+/->
```

Trade events log as they happen:
```
[ENTRY] ticket=123456 LONG 0.01 lots @ 3452.80 grade=A ob=M5 kz=london_kz SL=... TP=...
[EXIT]  ticket=123456 reason=TP1
```

Pipe to a log file for persistence:
```bash
slytrade live --symbol XAUUSDm --risk-cap 0.01 --live 2>&1 | tee -a live.log
```

Then `tail -f live.log` from another terminal.

## 6. Kill switch

- Ctrl+C the live terminal (positions already open stay open on MT5 — bot stops managing but broker-side SL/TP are already set, so they'll close on their own).
- To flatten everything: close MT5 manually or use the MT5 terminal one-click close.

## 7. Known limitations (before Layer 6 RL)

- No web dashboard yet (planned); stdout + log file for now
- No partial/ladder exits — one-shot (tp1=0.85R) per champion (this is what tested best PF 2.00)
- No BE lock or trailing stop in the live loop (broker-side SL/TP are static; champion PF validated without BE/trail)
- Single-symbol for now (we only trained on XAUUSDm)
- Weekend / closed-market: MT5 will return stale quotes; bot prints them and won't open trades if spread is unreasonable
- Signals use `champion_persona()` — `rl_training_persona()` is exported for Layer 6 training, not live

## Files
- `src/slytrade/live/trader.py` — main loop (LiveTrader class + CLI entry)
- `src/slytrade/live/__init__.py` — package marker
- CLI wired as `slytrade live` via `src/slytrade/cli.py`
- `start_mt5_bridge.sh` — launches Wine + MT5 + RPyC bridge
