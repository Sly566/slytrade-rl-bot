"""Structured logging for the runtime.

Two handlers are always available:

* ``console`` — human-readable logs (coloured when attached to a TTY).
* ``json`` — machine-readable JSON lines written to ``logs/slytrade.jsonl``
  with rotation, so Kubernetes/Fluentd/Datadog can ingest execution and audit
  events without parsing pretty-printed text.

Every module in :mod:`slytrade.runtime` shares the ``slytrade`` logger, so log
records are consistent and searchable by ``event`` / ``symbol`` / ``order_id``.
"""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Optional structured fields surfaced from log calls via the `extra=` kwarg.
_EXTRA_FIELDS = (
    "event",
    "symbol",
    "order_id",
    "status",
    "stage",
    "reason",
    "equity",
    "drawdown",
    "bars",
    "trades",
)


class JsonLineFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable console format."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s %(message)s", "%H:%M:%S")


def setup_logging(level: str = "INFO", log_dir: str = "logs", json_logs: bool = True) -> logging.Logger:
    """Configure the shared ``slytrade`` logger and return it.

    Safe to call more than once (idempotent): existing handlers are removed
    before re-adding so repeated setup never duplicates output.
    """
    logger = logging.getLogger("slytrade")
    logger.setLevel(level.upper())
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter())
    console.setLevel(level.upper())
    logger.addHandler(console)

    if json_logs:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path / "slytrade.jsonl",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonLineFormatter())
        file_handler.setLevel(level.upper())
        logger.addHandler(file_handler)

    return logger
