#!/bin/bash
# SlyTrade multi-symbol launcher — run all 9 pairs simultaneously
# Usage: bash run_all.sh

SYMBOLS=(
    "XAUUSDm"
    "BTCUSDm"
    "US30m"
    "USDJPYm"
    "USOILm"
    "EURUSDm"
    "XAGUSDm"
    "USTECm"
    "DE30m"
)

COMMON="--risk-cap 0.05 --working-lot 0.04 --max-open 3 --all --verbose --live"

for sym in "${SYMBOLS[@]}"; do
    echo "Starting $sym ..."
    slytrade live --symbol "$sym" $COMMON > "logs_${sym}.txt" 2>&1 &
    echo "  → PID $!  logs: logs_${sym}.txt"
done

echo ""
echo "All 9 symbols launched. Tail logs with:"
echo "  tail -f logs_XAUUSDm.txt"
echo ""
echo "Stop all with:"
echo "  pkill -f 'slytrade live'"
