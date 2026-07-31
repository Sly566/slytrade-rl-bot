# Phase 12 — Real Data Diagnostics and Bar-Close Alignment

This phase adds the safeguards needed before trusting real MT5 data in backtests.

## Linux MT5 bridge

On the Linux machine that runs MT5 through Wine/mt5linux, start your bridge first. A typical flow is:

```bash
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x16 &
wine "C:\\Program Files\\MetaTrader 5\\terminal64.exe" &
sleep 15
export WINEPREFIX="$HOME/.wine_mt5"
wine python -m mt5linux
```

Then in another terminal, from this repository:

```bash
python -m pip install -e ".[dev,data,mt5]"
python -m slytrade.cli mt5-info
python -m slytrade.cli resolve-symbols --contains XAU
```

## Bar-close alignment

MT5 bars are normally timestamped at bar open.

For an M1 bar at `08:00:00`, the OHLC values are not known until `08:01:00`.

Therefore, if a strategy uses the completed bar, its causal decision time is:

```text
decision_time = bar_open_time + timeframe_duration
```

The tick-aware backtester now executes bar-based signals using this causal decision time.

## Data inspection

Use `inspect-data` before trusting a backtest:

```bash
python -m slytrade.cli inspect-data \
  --bars-file data/raw/mt5_bars/symbol=XAUUSDm/timeframe=M1/year=2026/month=07/day=29.parquet \
  --ticks-file data/raw/mt5_ticks/symbol=XAUUSDm/year=2026/month=07/day=29.parquet \
  --timeframe M1
```

It reports:

- bar count and time range,
- bar decision-time range,
- duplicate/invalid bars,
- tick count and time range,
- spread statistics,
- tick coverage before each bar decision time.

## Why this matters

Without bar-close alignment, a backtest can accidentally use a candle's high/low/close before the candle is actually complete. That is lookahead bias.
