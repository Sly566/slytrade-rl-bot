# Multi-Symbol & Live-Fidelity Upgrade — the path to profit, end to end

This round closes the gap between the **validated** champion and what the bot
actually does **live**, and adds the machinery to scale across symbols.

## The problem that was found (and fixed)

The champion was validated at **+38.3% net** on the user's MT5 2-year run with
a **H4-trend alignment gate** as the core edge. But the live paper loop:

1. defaulted to **M1** — the timeframe validated to be **structurally losing**
   (round-trip cost ≈ 17.6% of R per trade; the cost wall);
2. only computed **single-timeframe** features, so the strategy's
   `require_mtf_alignment` gate found no higher-timeframe data and silently
   **skipped the H4-trend gate** — live was trading a strategy that was never
   validated.

**So the bot's most profitable, validated behaviour was not what ran live.**

## Fixes shipped (all tested on the real Exness archive)

1. **Live loop now trades the champion.** `RuntimeSettings.timeframe` defaults
   to **M15** (the +38.3% profile), and the loop resamples its rolling window
   to H4/D1 and merges `htf_*` + `mtf_bias`/`mtf_confluence_score`, so the
   H4-trend alignment gate fires live exactly as in backtest.
2. **Warm start.** The loop seeds its feature window from a stored bars file
   (`replay_bars_file` / `--history-bars`), so signals are warm from the first
   streamed bar instead of cold for the first N bars.
3. **Live-loop fidelity proof** (real M15, 48k bars):
   `htf_h4_bos_dir` **exact-match 100%**, `htf_h4_trend_strength` sign-match
   100%, and the **H4-trend ALIGNMENT decision agrees 100%** between the
   full-history backtest context and the live windowed context. The live loop
   makes the same calls as the validated system.
4. **Multi-symbol admission gate** (`slytrade admit XAUUSD,EURUSD,GBPUSD`):
   runs collect → align → champion backtest per symbol and admits only
   net-profitable symbols (PF ≥ 1, sane drawdown, ≥ 30 trades). A symbol must
   earn its place — same discipline as the RL promotion gate. Results land in
   `state/admitted_symbols.json`.
5. **Portfolio-level circuit breaker**: multi-symbol paper now shares a
   `PortfolioBreaker` that halts the whole book on an aggregate drawdown
   (daily + total), so a correlated multi-symbol loss can never blow past the
   portfolio risk budget.
6. **Incremental / online collection** (`slytrade collect_incremental`): pulls
   only the trailing window and merges into the archive (storage de-dupes), so
   the bot always runs on the freshest data without re-downloading history.
   Run it on a schedule or right before the paper loop.

## Cost reality check (Exness, 2026)

The champion is validated at **$3.5/lot commission + 5pt slippage**. Exness
account types (from Exness's own 2026 pages):

| Account | XAUUSD spread | Commission | Round-trip / lot |
|---|---|---|---|
| Standard | 20–35 pips | none | ~$20–35 |
| Raw Spread | ~0–5 pips | $3.50/side | ~$7 + spread |
| Zero | ~0 pips | $5.50/side (gold) | ~$11 |

The user's `MT5Trial9` terminal is a trial/Standard-type account, and the
validated assumptions are **conservative for a Raw/Zero account** and
**optimistic for Standard**. The champion's M15 edge (net +0.114R/trade after
$3.5+5pt) stays profitable on Raw/Zero; on a 20–35 pip Standard spread it does
**not** (round-trip cost would be ~3–5× the validated assumption). **Before
demo/live, confirm the real account spread/commission** with
`slytrade mt5-info` and set `configs/risk.yaml` `costs` accordingly — this is
the single highest-leverage remaining risk item.

## Why this is the honest path to profitability

- The **champion rule** is the profitable system today (+38.3% net on MT5 data).
- The **RL "superbrain"** is correctly implemented but cannot beat the champion
  on 667 trades of a +0.13R/trade edge (measured exhaustively — see
  `docs/RL_CALIBRATION_FINDINGS.md`); it stays behind the promotion gate.
- The levers that genuinely move the P&L are now in place: **trade the
  validated system live (not the losing M1 path)**, **spread risk across
  admitted symbols**, **cap correlated drawdowns at the portfolio level**, and
  **keep the data always fresh**.

## Next steps (in order)

1. `slytrade mt5-info` — confirm the real account spread/commission; set
   `configs/risk.yaml` `costs` to match.
2. `slytrade collect` + `slytrade paper --timeframe M15` — soak the champion
   live (paper) with the H4 gate now actually active.
3. `slytrade admit XAUUSD,EURUSD,GBPUSD,...` — validate each new symbol before
   it trades.
4. `slytrade paper-multi` on the admitted watchlist with the portfolio breaker.
5. Schedule `slytrade collect_incremental` (cron) so the archive is always
   current.
