#!/bin/sh
# SlyTrade container entrypoint — fail-closed startup guard.
#
# The image must never silently enable live trading. This guard refuses to boot
# if the operator has flipped SLYTRADE_ALLOW_LIVE without the corresponding
# deployment stage.
set -e

if [ "${SLYTRADE_ALLOW_LIVE:-0}" = "1" ]; then
  if [ "${SLYTRADE_STAGE:-paper}" != "demo" ]; then
    echo "Refusing to start: SLYTRADE_ALLOW_LIVE=1 requires SLYTRADE_STAGE=demo" >&2
    exit 1
  fi
  echo "WARNING: live trading is enabled for stage=${SLYTRADE_STAGE}" >&2
fi

exec python -m slytrade.cli "$@"
