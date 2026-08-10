from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, enums and datetimes into JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


class JsonlJournal:
    """Append-only JSONL audit journal.

    The journal is deliberately simple: every state-changing execution event can
    be replayed or inspected later. This is not a database replacement; it is an
    auditable foundation we can later back with SQLite/Postgres.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {"event_type": event_type, **to_jsonable(payload)}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows


class SqliteJournal:
    """Durable, transactional append-only event journal.

    SQLite is used deliberately here because it is available in the standard
    library, survives process restarts, and gives each event an ordering
    sequence that can be used for deterministic state reconstruction.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        row = json.dumps(to_jsonable(payload), sort_keys=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO execution_events (event_type, payload, created_at) VALUES (?, ?, ?)",
                (event_type, row, datetime.now().astimezone().isoformat()),
            )
            connection.commit()

    def read_all(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT event_type, payload FROM execution_events ORDER BY sequence"
            ).fetchall()
        return [{"event_type": event_type, **json.loads(payload)} for event_type, payload in rows]
