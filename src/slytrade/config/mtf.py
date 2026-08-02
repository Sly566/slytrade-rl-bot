"""Multi-timeframe configuration (single source of truth)."""
from dataclasses import dataclass, field


@dataclass
class MTFConfig:
    execution_timeframe: str = "M1"
    target_timeframe: str = "M5"
    intraday_timeframe: str = "M15"

    # Dynamic MTF scoring parameters (no magic numbers)
    min_mtf_score: int = 2
    require_mtf_bias_alignment: bool = True

    observe_timeframes: list[str] = field(default_factory=lambda: 
        ["M1", "M5", "M15", "H1", "H4", "D1", "W1"])

    primary_symbols: list[str] = field(default_factory=lambda: ["XAUUSD"])
    secondary_symbols: list[str] = field(default_factory=lambda: ["BTCUSD"])

    use_ticks: bool = True
    use_bars: bool = True
    tick_source: str = "MT5"
    bar_source: str = "MT5"

    asset_classes: dict[str, list[str]] = field(default_factory=lambda: {
        "metals": ["XAUUSD", "XAGUSD"],
        "crypto": ["BTCUSD", "ETHUSD"],
        "forex": ["EURUSD", "GBPUSD", "USDJPY"],
        "indices": ["US500", "US30", "NAS100", "DE30"],
        "commodities": ["USOIL"],
    })

DEFAULT_MTF = MTFConfig()

TIMEFRAME_TO_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080
}

def get_higher_timeframes(execution_tf: str = "M1") -> list[str]:
    exec_min = TIMEFRAME_TO_MINUTES[execution_tf]
    return [tf for tf in DEFAULT_MTF.observe_timeframes 
            if TIMEFRAME_TO_MINUTES.get(tf, 0) >= exec_min]

def is_valid_timeframe(tf: str) -> bool:
    return tf in TIMEFRAME_TO_MINUTES
