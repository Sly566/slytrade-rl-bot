"""Account-currency handling for non-USD trading accounts.

The sizing/PnL engine assumes USD point values. A ZAR-denominated account (like
the Exness demo) needs its equity converted to USD before risk-based sizing, and
back before display. This module resolves the conversion rate from the live MT5
terminal (USD<CUR> or <CUR>USD symbol) and falls back to a configured rate when
the terminal cannot supply one.
"""

from __future__ import annotations

from typing import Any


class CurrencyConverter:
    """Resolve account currency → USD rates with a live terminal + fallback."""

    def __init__(self, fallback_rate: float = 1.0):
        if fallback_rate <= 0:
            raise ValueError("fallback_rate must be positive")
        self.fallback_rate = fallback_rate
        self._rate: float | None = None
        self._currency: str = "USD"

    def resolve(self, mt5: Any, account_currency: str) -> float:
        """Return USD-per-account-unit rate (i.e. 1 account unit = X USD).

        * account currency USD -> 1.0
        * USD<CUR> pair (e.g. USDZAR): 1 unit CUR = 1/mid USD
        * <CUR>USD pair (e.g. ZARUSD): 1 unit CUR = mid USD
        * otherwise the configured fallback.
        """
        currency = (account_currency or "USD").upper()
        self._currency = currency
        if currency == "USD":
            self._rate = 1.0
            return 1.0

        candidates = [f"USD{currency}", f"{currency}USD"]
        for pair in candidates:
            tick = None
            try:
                if hasattr(mt5, "symbol_info_tick"):
                    tick = mt5.symbol_info_tick(pair)
            except Exception:  # pragma: no cover - broker dependent
                tick = None
            if tick is None:
                continue
            bid = float(getattr(tick, "bid", 0.0) or 0.0)
            ask = float(getattr(tick, "ask", 0.0) or 0.0)
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            if pair == f"USD{currency}":
                self._rate = 1.0 / mid  # USDZAR mid = ZAR per USD
            else:
                self._rate = mid  # ZARUSD mid = USD per ZAR
            return self._rate

        self._rate = self.fallback_rate
        return self._rate

    def to_usd(self, amount: float, mt5: Any | None = None, account_currency: str | None = None) -> float:
        if mt5 is not None and account_currency:
            rate = self.resolve(mt5, account_currency)
        else:
            rate = self._rate or self.fallback_rate
        return amount * rate

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def rate(self) -> float:
        return self._rate or self.fallback_rate


def load_converter(risk_config: dict) -> CurrencyConverter:
    currency = risk_config.get("currency", {}) or {}
    costs = risk_config.get("costs", {})
    rate = float(currency.get("rate_to_usd") or costs.get("currency_rate_to_usd", 1.0) or 1.0)
    return CurrencyConverter(fallback_rate=rate)
