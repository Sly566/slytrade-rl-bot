"""Symbol specification — dynamic per-asset contract math.

In live trading this comes straight from `mt5.symbol_info()`. In backtest
we either load persisted symbol info (recommended) or fall back to sensible
defaults for the asset class.

Hardcoding pip/contract sizes for XAUUSD is forbidden by Sly's spec — all
math must derive from symbol properties so the same engine runs metals,
forex, crypto, indices, energies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Asset-class defaults (used only when no live/persisted info is available)
# --------------------------------------------------------------------------- #
_ASSET_DEFAULTS: Dict[str, dict] = {
    # Metals: standard 100-oz contract, tick size 0.01 (2-digit) or 0.001 (3-digit raw)
    "XAU": {"contract_size": 100.0, "point": 0.001, "digits": 3},
    "XAG": {"contract_size": 5000.0, "point": 0.001, "digits": 3},
    # Forex majors
    "EUR": {"contract_size": 100_000.0, "point": 0.00001, "digits": 5},
    "GBP": {"contract_size": 100_000.0, "point": 0.00001, "digits": 5},
    "USD": {"contract_size": 100_000.0, "point": 0.00001, "digits": 5},
    "JPY": {"contract_size": 100_000.0, "point": 0.001, "digits": 3},
    # Crypto
    "BTC": {"contract_size": 1.0, "point": 0.01, "digits": 2},
    "ETH": {"contract_size": 1.0, "point": 0.01, "digits": 2},
    # Indices
    "USTEC": {"contract_size": 10.0, "point": 0.1, "digits": 1},
    "US500": {"contract_size": 10.0, "point": 0.1, "digits": 1},
    # Energies
    "WTI":  {"contract_size": 100.0, "point": 0.01, "digits": 2},
    "XBR":  {"contract_size": 100.0, "point": 0.01, "digits": 2},
}


@dataclass(frozen=True)
class SymbolSpec:
    """Per-asset contract specification (mirrors mt5.symbol_info core fields)."""

    name: str                               # e.g. "XAUUSDm"
    base: str = "XAU"                       # asset-class family for defaults
    currency_profit: str = "USD"            # quote currency
    point: float = 0.001                    # smallest tick increment in price
    digits: int = 3                         # decimal digits in quoted price
    contract_size: float = 100.0            # units per 1.0 lot
    volume_min: float = 0.01                # min lot
    volume_max: float = 100.0               # max lot
    volume_step: float = 0.01               # lot step
    # Per-lot value of a one-point move in quote currency.
    # For XAUUSD: 1 point = $0.001 move × 100 oz = $0.10 per point per lot.
    # For EURUSD: 1 point = $0.00001 × 100_000 = $1.00 per point per lot.
    tick_value: float = 0.10
    tick_size: float = 0.001                # price increment == point (for most brokers)

    def price_to_points(self, price_diff: float) -> float:
        """Convert a price difference (e.g. stop distance) into MT5 points."""
        return price_diff / self.point if self.point else 0.0

    def profit_per_lot(self, price_diff: float) -> float:
        """Profit in `currency_profit` for 1 lot when price moves `price_diff`."""
        points = self.price_to_points(price_diff)
        return points * self.tick_value

    def lots_for_risk(self, price_risk: float, account_risk_ccy: float) -> float:
        """Return lot size risking `account_risk_ccy` in quote currency
        for a stop `price_risk` away."""
        if price_risk <= 0:
            return 0.0
        pp_lot = self.profit_per_lot(price_risk)  # ccy per lot per stop
        if pp_lot <= 0:
            return 0.0
        lots = account_risk_ccy / pp_lot
        # Round to volume_step
        lots = np.floor(lots / self.volume_step) * self.volume_step
        return float(np.clip(lots, self.volume_min, self.volume_max))


def _infer_base(symbol: str) -> str:
    """Strip broker suffix ('m','.a','!', etc.) and return first 3 chars as family."""
    base = symbol.rstrip("mabcdefghijklnopqrstuvwxyz!._-")
    return base[:3].upper()


def spec_for_symbol(symbol: str, overrides: Optional[Dict] = None) -> SymbolSpec:
    """Build a SymbolSpec using defaults inferred from symbol name.

    `overrides` can inject live MT5 fields (point/contract_size/digits/etc.).
    """
    base = _infer_base(symbol)
    d = _ASSET_DEFAULTS.get(base, _ASSET_DEFAULTS["XAU"]).copy()
    if overrides:
        d.update(overrides)
    # Derive tick_value from contract_size/point for forex/metals:
    #   1-point move × contract_size in quote ccy
    if "tick_value" not in (overrides or {}):
        d["tick_value"] = d["contract_size"] * d["point"]
    return SymbolSpec(name=symbol, base=base, **d)


@dataclass
class AccountSpec:
    """Account-level parameters for the backtester."""

    starting_equity: float = 2000.0     # in account currency
    currency: str = "ZAR"               # ZAR demo per the Exness account
    leverage: int = 2000                # used for margin check only
    # Conversion: profit_currency -> account_currency.
    # For ZAR accounts trading XAUUSD (profit in USD), we need USD/ZAR.
    fx_to_account: Dict[str, float] = field(default_factory=lambda: {"USD": 18.5})
    commission_per_lot_rt: float = 0.0  # Exness zero/standard: 0; raw: ~$3/lot/side
    # Slippage in points applied on stop/limit fills (guesstimated)
    slippage_points: int = 5

    def to_account_ccy(self, amount_quote: float, quote_ccy: str) -> float:
        if quote_ccy == self.currency:
            return amount_quote
        rate = self.fx_to_account.get(quote_ccy, 1.0)
        return amount_quote * rate
