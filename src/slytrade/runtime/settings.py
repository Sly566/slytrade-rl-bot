"""Fail-closed runtime settings.

Every operational value is driven by environment variables (prefixed with
``SLYTRADE_``) or by the YAML config files. There are deliberately **no**
hard-coded operational thresholds: defaults are safe, and the live-trading flag
can never be turned on by accident because it is read from the environment and
cross-checked against the deployment gate.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingStage(StrEnum):
    """Operational stage of the running process.

    Ordered by increasing blast radius. The paper loop refuses to start for any
    stage other than DRY_RUN / PAPER / SHADOW unless the operator has explicitly
    approved live execution through the deployment gate.
    """

    DRY_RUN = "dry_run"
    PAPER = "paper"
    SHADOW = "shadow"
    DEMO = "demo"


class RuntimeSettings(BaseSettings):
    """Runtime configuration loaded from environment variables / .env.

    Example::

        SLYTRADE_ALLOW_LIVE=0 SLYTRADE_SYMBOL=XAUUSD slytrade paper
    """

    model_config = SettingsConfigDict(
        env_prefix="SLYTRADE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Identity / safety -------------------------------------------------
    env: str = "development"
    allow_live: bool = False
    stage: TradingStage = TradingStage.PAPER

    # --- Observability ------------------------------------------------------
    metrics_port: int = 9108
    metrics_bind: str = "0.0.0.0"
    metrics_enabled: bool = True
    log_level: str = "INFO"
    log_dir: str = "logs"
    json_logs: bool = True

    # --- State / storage ----------------------------------------------------
    config_dir: str = "configs"
    state_dir: str = "state"
    data_dir: str = "data"
    kill_switch_path: str = "state/kill-switch.json"

    # --- Paper loop ----------------------------------------------------------
    symbol: str = "XAUUSD"
    timeframe: str = "M1"
    strategy: str = "persona-adaptive"
    initial_balance: float = 100_000.0
    poll_seconds: float = 1.0
    stale_quote_seconds: float = 5.0
    heartbeat_interval_seconds: float = 5.0
    max_runtime_seconds: float = 0.0  # 0 = run until stopped
    default_spread_points: float = 20.0
    symbol_spec_file: str | None = None
    replay_ticks_file: str | None = None
    replay_bars_file: str | None = None

    # --- Trading window (UTC) -------------------------------------------------
    trading_days: str = "mon,tue,wed,thu,fri"
    trading_start_utc: str = "00:00"
    trading_end_utc: str = "23:59"

    # --- Alerting (all empty/disabled by default) ------------------------------
    alert_webhook_url: str = ""
    alert_telegram_bot_token: str = ""
    alert_telegram_chat_id: str = ""

    # --- Red-folder news gate ---------------------------------------------------
    news_config_file: str = "configs/news.yaml"
    news_enabled: bool = False
    # Economic-calendar feed (GAP-4): optional JSON/CSV file or JSON URL. When
    # set and news_enabled, the red-folder gate is built from this feed.
    calendar_path: str = ""
    calendar_url: str = ""

    def fail_closed_checks(self) -> list[str]:
        """Return a list of configuration problems that must block startup.

        The paper loop calls this before doing anything else so a misconfigured
        container fails loudly instead of silently running with defaults.
        """
        problems: list[str] = []
        if self.metrics_enabled and not (0 < self.metrics_port < 65536):
            problems.append(f"metrics_port out of range: {self.metrics_port}")
        if self.poll_seconds < 0:
            problems.append("poll_seconds cannot be negative")
        if self.initial_balance <= 0:
            problems.append("initial_balance must be positive")
        if not self.symbol.strip():
            problems.append("symbol cannot be empty")
        if self.allow_live and self.stage not in (TradingStage.DEMO,):
            problems.append("SLYTRADE_ALLOW_LIVE=1 requires stage=demo and the deployment gate")
        return problems

    @property
    def trading_days_set(self) -> frozenset[str]:
        return frozenset(day.strip().lower() for day in self.trading_days.split(",") if day.strip())

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir)

    @property
    def config_path(self) -> Path:
        return Path(self.config_dir)


def runtime_settings() -> RuntimeSettings:
    """Build the process-wide settings (cached per call site is not needed)."""
    return RuntimeSettings()
