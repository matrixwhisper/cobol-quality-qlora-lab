"""Strict JSONL ingestion and provenance-preserving serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from data_quality.schema import CobolTask, TaskFormatError


def load_tasks(path: str | Path) -> list[CobolTask]:
    source = Path(path)
    tasks: list[CobolTask] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TaskFormatError("each JSONL row must be an object")
                task = CobolTask.from_dict(value)
                if task.task_id in seen:
                    raise TaskFormatError(f"duplicate task_id {task.task_id!r}")
                seen.add(task.task_id)
                tasks.append(task)
            except (json.JSONDecodeError, TaskFormatError) as exc:
                raise TaskFormatError(f"{source}:{line_number}: {exc}") from exc
    if not tasks:
        raise TaskFormatError(f"{source} contains no tasks")
    return tasks


def write_tasks(tasks: Iterable[CobolTask], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    task_list = list(tasks)
    if not task_list:
        raise ValueError("refusing to write an empty task collection")
    destination.write_text(
        "".join(json.dumps(task.to_dict(), ensure_ascii=False) + "\n" for task in task_list),
        encoding="utf-8",
    )
    return destination