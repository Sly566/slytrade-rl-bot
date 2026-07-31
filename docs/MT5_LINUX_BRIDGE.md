# MT5 Linux Bridge and Symbol Resolution

SlyTrade supports base symbols like `XAUUSD` at the CLI. The bot resolves them to the broker's actual MT5 symbol before requesting ticks or bars.

Examples:

```text
XAUUSD -> XAUUSDm
BTCUSD -> BTCUSDm
```

## Symbol resolution commands

```bash
python -m slytrade.cli resolve-symbols --contains XAUUSD
python -m slytrade.cli resolve-symbols --contains XAU
python -m slytrade.cli resolve-symbols --contains BTC
```

## Data collection with base symbols

These commands may be called with `XAUUSD`; the collector will resolve to the actual broker symbol automatically:

```bash
python -m slytrade.cli collect-bars \
  --symbol XAUUSD \
  --timeframe M1 \
  --start 2026-07-29T08:00:00 \
  --end 2026-07-29T09:00:00
```

```bash
python -m slytrade.cli collect-ticks \
  --symbol XAUUSD \
  --start 2026-07-29T08:00:00 \
  --end 2026-07-29T09:00:00
```

Use `--no-resolve` only when you intentionally want to bypass symbol resolution.

## RPyC invalid message type

If you see an error like:

```text
ValueError: invalid message type: 18
```

then the Linux Python client connected to a stale or incompatible Wine-side RPyC server.

Recommended fix:

1. Stop old bridge processes:

```bash
pkill -f mt5linux 2>/dev/null || true
pkill -f rpyc 2>/dev/null || true
pkill -f "wine.*python" 2>/dev/null || true
```

2. Align versions in Wine Python:

```bash
export WINEPREFIX="$HOME/.wine_mt5"
wine python -m pip install --upgrade --force-reinstall mt5linux==1.0.11 rpyc==6.0.2 plumbum==1.7.0 pyparsing==3.3.2 MetaTrader5
```

3. Restart MT5 and the bridge:

```bash
export DISPLAY=:99
export WINEPREFIX="$HOME/.wine_mt5"
Xvfb :99 -screen 0 1024x768x16 &
wine "C:\\Program Files\\MetaTrader 5\\terminal64.exe" &
sleep 15
wine python -m mt5linux
```

4. In another terminal:

```bash
python -m slytrade.cli mt5-info
```
