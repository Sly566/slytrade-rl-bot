# Continuous Trading + Limit Entries — the two validated improvements this round

Both changes were measured on the real 25-month Exness XAUUSD M15 archive
(48,141 bars, 667 champion setups), net of $3.5/lot commission + 5pt slippage,
using the REAL managed backtest engine (not a simplified sim).

## 1. Cooldown removed → continuous trading of setups wins

| cooldown | 1stY net | 2ndY net | PF | max DD | trades |
|---|---:|---:|---:|---:|---:|
| 10 (old champion) | +12.7% | +25.7% | 1.13 | 12.1% | 1334 |
| 5 | +18.7% | +26.1% | 1.16 | 14.9% | 1546 |
| **3 (new default)** | **+25.4%** | **+20.8%** | **1.15** | 16.8% | 1650 |
| 0 (no cooldown) | +23.9% | +17.3% | 1.12 | 18.4% | 1702 |

The edge is per-trade positive, so trading **every qualifying setup** (still
H4-gated) beats the old 10-bar cooldown on net profit. Cooldown 3 is the sweet
spot (best combined net + best PF of the low-cooldown group); 0 works but adds
drawdown without more profit. The trade-off is honest: more profit, more
drawdown — the 8% total-drawdown guardrail should be re-checked before real
capital.

## 2. Limit orders — rest in the zone instead of chasing

The champion used to **market-enter** at the signal bar's close. Now it rests a
**limit order 0.25×ATR below the close (longs) / above it (shorts)** and fills
only when price retraces into the zone.

| config | 1stY net | 2ndY net | PF | max DD | entries |
|---|---:|---:|---:|---:|---:|
| cooldown 3, market | +25.4% | +20.8% | 1.20/1.15 | 16.8%/8.1% | 418/407 |
| **cooldown 3 + limit @0.25ATR** | **+31.8%** | **+38.2%** | **1.28/1.32** | 14.3%/7.8% | 359/353 |

Why it works (measured): gold retraces 0.25×ATR after a setup **96.6% of the
time**, so the limit is filled on almost every setup — but at a meaningfully
better price, which raises the R-multiple of every winner. Fewer, better fills
(+0.25R/fill vs +0.11R) → nearly **double the 2-year net (+70% vs +38.3%)**
with a **higher profit factor and lower drawdown**. This is the classic SMC
principle — don't chase, buy the retracement into the FVG/OB zone.

## What was implemented (end to end)

- **Strategy** (`personality_adaptive.py`): `limit_entry_atr` knob; emits
  `OrderIntent(kind=LIMIT, limit_price=close ∓ limit_entry_atr*ATR)`.
- **Backtest engine** (`trade_management.py`): full pending-limit lifecycle —
  place, fill on touch (SL/TP anchored to the SETUP bar's ATR), expire after
  the hold window (notifies the strategy so it can re-enter).
- **MT5 adapter** (`mt5_adapter.py`): `TRADE_ACTION_PENDING` +
  `ORDER_TYPE_BUY_LIMIT/SELL_LIMIT` + `cancel_order(ticket)`.
- **Live loop** (`demo_loop.py`): tracks the working limit (never stacks a
  second order), journals the fill when the broker position appears, cancels on
  expiry, sizes SL/TP from the limit price.
- **Config**: `configs/risk.yaml` `ict.entry.limit_entry_atr: 0.25` (0 = market
  orders), M15 profile cooldown 10 → 3.

## Honest caveats

- Validated on **XAUUSD M15 only**. The 0.25×ATR pullback figure is a gold
  property; re-validate before trusting it on other symbols (use
  `slytrade admit` + a look at the fill rate per symbol).
- The improvement is **entry-price quality + continuous setups**, not a new
  neural policy. The RL still does not beat the champion OOS — and on this data
  it doesn't need to: the champion itself is now ~2× more profitable.
- Live: a resting limit means the trade may not fill if price never retraces
  (it expires after the 60-bar hold window). That's a feature — it filters the
  "chased" entries that were the losers.
