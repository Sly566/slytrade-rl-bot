"""Operator alerting for the trading runtime.

Delivers operational alerts (kill switch, soak anomalies, broker errors,
shutdown summaries) to any combination of:

* the structured logger (always on),
* a generic JSON webhook (Slack/Teams/Discord-compatible),
* a Telegram bot.

Every transport is **best-effort**: a down webhook or a missing network must
never take the trading loop down. Sending failures are logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger("slytrade.alerts")


@dataclass(frozen=True)
class Alert:
    severity: str  # info | warning | critical
    title: str
    detail: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class AlertChannel:
    """Sink for operational alerts. Implementations must never raise."""

    def send(self, alert: Alert) -> bool:
        raise NotImplementedError


class LogChannel(AlertChannel):
    """Always-available channel that routes to the structured logger."""

    def __init__(self, target_logger: logging.Logger | None = None) -> None:
        self._logger = target_logger or logger

    def send(self, alert: Alert) -> bool:
        self._logger.log(
            {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}.get(
                alert.severity, logging.INFO
            ),
            "ALERT [%s] %s: %s",
            alert.severity,
            alert.title,
            alert.detail,
            extra={"event": "alert", "status": alert.severity, "reason": alert.title},
        )
        return True


class WebhookChannel(AlertChannel):
    """POST a JSON payload to a generic webhook URL."""

    def __init__(self, url: str, *, timeout: float = 5.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"webhook URL must be http(s), got {url!r}")
        self.url = url
        self.timeout = timeout

    def send(self, alert: Alert) -> bool:
        payload = json.dumps(asdict(alert), sort_keys=True).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("webhook delivery failed: %s", exc, extra={"event": "alert_failed"})
            return False


class TelegramChannel(AlertChannel):
    """Send a plain-text message through the Telegram Bot API."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        *,
        timeout: float = 5.0,
        base_url: str = "https://api.telegram.org",
    ) -> None:
        if not bot_token or not chat_id:
            raise ValueError("bot_token and chat_id are required")
        self.url = f"{base_url}/bot{bot_token}/sendMessage"
        self.chat_id = chat_id
        self.timeout = timeout

    def send(self, alert: Alert) -> bool:
        text = f"[{alert.severity.upper()}] {alert.title}\n{alert.detail}".strip()
        query = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text})
        request = urllib.request.Request(f"{self.url}?{query}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("telegram delivery failed: %s", exc, extra={"event": "alert_failed"})
            return False


class AlertManager:
    """Fan an alert out to every configured channel without ever raising."""

    def __init__(self, channels: list[AlertChannel] | None = None) -> None:
        self.channels = channels or [LogChannel()]

    def alert(self, severity: str, title: str, detail: str = "") -> Alert:
        alert = Alert(severity=severity, title=title, detail=detail)
        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception:  # pragma: no cover - defensive fan-out
                logger.exception("alert channel failed", extra={"event": "alert_failed"})
        return alert

    @classmethod
    def from_settings(cls, settings, target_logger: logging.Logger | None = None) -> AlertManager:
        """Build an AlertManager from RuntimeSettings (empty URLs -> disabled)."""
        channels: list[AlertChannel] = [LogChannel(target_logger or logger)]
        if getattr(settings, "alert_webhook_url", ""):
            channels.append(WebhookChannel(settings.alert_webhook_url))
        if getattr(settings, "alert_telegram_bot_token", "") and getattr(settings, "alert_telegram_chat_id", ""):
            channels.append(
                TelegramChannel(settings.alert_telegram_bot_token, settings.alert_telegram_chat_id)
            )
        return cls(channels)
