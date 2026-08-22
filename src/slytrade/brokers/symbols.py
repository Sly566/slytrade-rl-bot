"""Broker symbol suffix handling and variant discovery.

Brokers (Exness in particular) append suffixes to symbol names:
- XAUUSDm (micro/cent account suffix 'm')
- XAUUSD.a, XAUUSD.r, XAUUSD! (execution-type suffixes)
- XAUUSD-micro

This module generates all possible variants and locates the actual raw
symbol names on the connected MT5 terminal.
"""
from __future__ import annotations

from typing import List, Optional

from . import BROKER_SUFFIXES  # re-export from config for convenience


def symbol_variants(base: str) -> List[str]:
    """Return all possible broker-suffixed variants for a base symbol."""
    seen = {}
    for suffix in BROKER_SUFFIXES:
        for variant in (f"{base}{suffix}", f"{base.upper()}{suffix}"):
            seen[variant] = None
    return list(seen.keys())


def detect_raw_symbol(mt5, base: str, prefer: Optional[str] = None) -> Optional[str]:
    """Find the actual symbol name on the connected MT5 terminal.

    If `prefer` is given, checks that first. Otherwise checks all variants
    in order and returns the first one visible and selectable in MT5.
    Returns None if no variant is found.
    """
    candidates: List[str] = []
    if prefer:
        candidates.append(prefer)
    candidates.extend(symbol_variants(base))

    for sym in candidates:
        info = mt5.symbol_info(sym)
        if info is not None and getattr(info, "visible", True):
            # Ensure the symbol is selected in MarketWatch
            if not info.visible:
                try:
                    mt5.symbol_select(sym, True)
                except Exception:
                    continue
            return sym
    return None
