#!/usr/bin/env bash
# SlyTrade — MT5 Linux bridge launcher (Wine + mt5linux)
#
# Fixes the two common startup failures:
#   1. wine "version mismatch 958/959"  -> stale wineserver / two wine installs
#   2. Xvfb "already active for display" -> reuse or clean the stale lock
#
# Usage:  bash start_mt5_bridge.sh
# Keep the terminal open — the mt5linux RPyC server runs in the foreground.
# Verify from another terminal:  python -m slytrade.cli mt5-info

set -euo pipefail

export WINEPREFIX="${WINEPREFIX:-$HOME/.wine_mt5}"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
export DISPLAY=":${DISPLAY_NUM}"

echo "== 1/6 Killing stale wine + bridge processes =="
wineserver -k 2>/dev/null || true
pkill -9 -f wineserver 2>/dev/null || true
pkill -9 -f terminal64 2>/dev/null || true
pkill -9 -f mt5linux 2>/dev/null || true
pkill -9 -f rpyc 2>/dev/null || true
pkill -9 -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
rm -f "/tmp/.X${DISPLAY_NUM}-lock"
sleep 2

echo "== 2/6 Checking wine =="
wine --version
if command -v wine64 >/dev/null 2>&1 && [ "$(command -v wine)" != "$(command -v wine64)" ]; then
  echo "WARNING: wine and wine64 resolve to different binaries — this causes version mismatch." >&2
  echo "         Unify them (see docs/MT5_LINUX_BRIDGE.md) or remove one wine install." >&2
fi

echo "== 3/6 Refreshing prefix =="
wineboot -u 2>/dev/null || true

echo "== 4/6 Installing wine-python bridge packages =="
wine python -m pip install -q --upgrade MetaTrader5 mt5linux==1.0.11 rpyc==6.0.2 plumbum==1.7.0 pyparsing==3.3.2 || echo "warn: wine python pip install failed"

echo "== 5/6 Starting Xvfb :${DISPLAY_NUM} =="
if pgrep -f "Xvfb :${DISPLAY_NUM}" >/dev/null 2>&1; then
  echo "Xvfb :${DISPLAY_NUM} already running — reusing it"
else
  rm -f "/tmp/.X${DISPLAY_NUM}-lock"
  Xvfb ":${DISPLAY_NUM}" -screen 0 1280x800x16 &
  sleep 2
fi

echo "== 6/6 Launching MT5 then the bridge (keep this terminal open) =="
wine "C:\\Program Files\\MetaTrader 5\\terminal64.exe" &
echo "Waiting 20s for the terminal to log in…"
sleep 20
echo "Starting mt5linux RPyC server on :18812 …"
exec wine python -m mt5linux
