"""Production runtime: configuration, observability and the paper-trading loop.

The modules in this package turn the research/backtest stack into a supervised
process that can run inside a container or on Kubernetes. Nothing here enables
live trading by itself: every entry point is fail-closed unless both the
``SLYTRADE_ALLOW_LIVE`` flag and the deployment gate are satisfied.
"""

from slytrade.runtime.settings import RuntimeSettings, TradingStage, runtime_settings

__all__ = ["RuntimeSettings", "TradingStage", "runtime_settings"]
