from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolResolution:
    requested: str
    resolved: str
    description: str = ""
    exact: bool = False


def _symbol_name(symbol: Any) -> str:
    if isinstance(symbol, dict):
        return str(symbol.get("name", symbol))
    return str(getattr(symbol, "name", symbol))


def _symbol_description(symbol: Any) -> str:
    if isinstance(symbol, dict):
        return str(symbol.get("description", ""))
    return str(getattr(symbol, "description", ""))


def get_all_symbol_names(mt5: Any) -> list[str]:
    symbols = _symbols_get(mt5)
    if symbols is None:
        return []
    return sorted(_symbol_name(symbol) for symbol in symbols)


def list_matching_symbols(mt5: Any, contains: str) -> list[SymbolResolution]:
    """List symbols whose name contains a case-insensitive text fragment."""
    fragment = contains.lower()
    symbols = _symbols_get(mt5)
    if symbols is None:
        return []
    matches: list[SymbolResolution] = []
    for symbol in symbols:
        name = _symbol_name(symbol)
        if fragment in name.lower():
            matches.append(
                SymbolResolution(
                    requested=contains,
                    resolved=name,
                    description=_symbol_description(symbol),
                    exact=name.lower() == fragment,
                )
            )
    return sorted(matches, key=lambda item: (not item.exact, len(item.resolved), item.resolved))


def resolve_symbol(mt5: Any, requested: str, *, select: bool = True) -> SymbolResolution:
    """Resolve a base symbol like XAUUSD to the broker's actual MT5 symbol.

    Resolution priority:
    1. exact case-insensitive match,
    2. symbol starts with requested base,
    3. symbol contains requested base,
    4. fail with a useful error.
    """
    requested_clean = requested.strip()
    if not requested_clean:
        raise ValueError("requested symbol cannot be empty")

    symbols = _symbols_get(mt5)
    if symbols is None:
        raise RuntimeError("mt5.symbols_get() returned None; cannot resolve symbols")

    candidates: list[tuple[int, int, str, Any]] = []
    requested_lower = requested_clean.lower()
    for symbol in symbols:
        name = _symbol_name(symbol)
        name_lower = name.lower()
        if name_lower == requested_lower:
            candidates.append((0, len(name), name, symbol))
        elif name_lower.startswith(requested_lower):
            candidates.append((1, len(name), name, symbol))
        elif requested_lower in name_lower:
            candidates.append((2, len(name), name, symbol))

    if not candidates:
        available_hint = ", ".join(get_all_symbol_names(mt5)[:20])
        raise ValueError(f"Could not resolve symbol '{requested_clean}'. First available symbols: {available_hint}")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    rank, _, resolved_name, raw_symbol = candidates[0]

    if select and hasattr(mt5, "symbol_select"):
        selected = mt5.symbol_select(resolved_name, True)
        if selected is False:
            raise RuntimeError(f"Resolved {requested_clean} -> {resolved_name}, but mt5.symbol_select returned False")

    return SymbolResolution(
        requested=requested_clean,
        resolved=resolved_name,
        description=_symbol_description(raw_symbol),
        exact=rank == 0,
    )


def _symbols_get(mt5: Any) -> list[Any]:
    """Read symbol names safely through mt5linux's remote interpreter.

    RPyC cannot pickle MetaTrader's dynamically-created ``SymbolInfo`` named
    tuples. The bridge can evaluate a projection to plain dictionaries/strings
    remotely, which preserves compatibility with the official local API.
    """
    try:
        symbols = mt5.symbols_get()
    except Exception:
        container = getattr(mt5, "_container", None)
        if container is None or not hasattr(container, "eval"):
            raise
        return list(container.eval("[s._asdict() for s in (mt5.symbols_get() or [])]"))
    return list(symbols or [])
