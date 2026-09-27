"""Append-only SQLite reviewer decision history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from data_quality.schema import CobolTask, ReviewStatus


class ReviewLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    reviewer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_task_idx ON events(task_id);
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def import_tasks(self, tasks: list[CobolTask]) -> int:
        inserted = 0
        with self._connect() as connection:
            for task in tasks:
                result = connection.execute(
                    "INSERT OR IGNORE INTO tasks(task_id, task_json) VALUES(?, ?)",
                    (
                        task.task_id,
                        json.dumps(task.to_dict(), sort_keys=True),
                    ),
                )
                inserted += result.rowcount
        return inserted

    def record(
        self,
        task_id: str,
        reviewer: str,
        status: ReviewStatus,
        rationale: str,
    ) -> None:
        if not reviewer.strip() or not rationale.strip():
            raise ValueError("reviewer and rationale are required")
        if status is ReviewStatus.UNREVIEWED:
            raise ValueError("cannot record an unreviewed event")
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone() is None:
                raise KeyError(task_id)
            connection.execute(
                """INSERT INTO events(task_id, reviewer, status, rationale, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    task_id,
                    reviewer,
                    status.value,
                    rationale,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def history(self, task_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT reviewer, status, rationale, created_at
                   FROM events WHERE task_id=? ORDER BY event_id""",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]