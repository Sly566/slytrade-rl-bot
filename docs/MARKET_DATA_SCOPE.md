# Market Data Scope

SlyTrade separates market data by source and by depth.

## Historical data supported now

```text
MT5 bars              OHLCV bars from the trading terminal
MT5 ticks             Recent bid/ask ticks from the trading terminal/bridge
Exness archive ticks  Historical bid/ask ticks from Exness public archive
```

## Level 1 vs Level 2

Historical Exness tick archives provide Level 1 data:

```text
timestamp, bid, ask
```

This is not true Level 2 order book depth.

True L2 would require order-book levels:

```text
bid level 1 price/volume
bid level 2 price/volume
ask level 1 price/volume
ask level 2 price/volume
...
```

Exness public tick history does not provide historical L2 depth. Therefore SlyTrade does not fabricate historical L2. The bot uses truthful Level 1 bid/ask ticks for historical execution realism.

## Live / forward L2

If MT5 `market_book_add` and `market_book_get` are available for the broker symbol, a future phase can record live L2 going forward. Until that capability is confirmed, historical research must not depend on L2.

## Data truth rule

The project must never label synthetic or inferred data as real market data. If a feature is derived from ticks, it must be named as a derived tick feature, not L2.
