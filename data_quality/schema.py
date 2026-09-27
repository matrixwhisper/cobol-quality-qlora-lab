"""Data contracts for reviewed COBOL test-generation tasks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class TaskFormatError(ValueError):
    """Raised when a COBOL benchmark task is malformed."""


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REVISED = "revised"
    REJECTED = "rejected"


@dataclass(frozen=True)
class CobolCase:
    case_id: str
    stdin: str
    expected_stdout: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CobolCase":
        required = ("case_id", "stdin", "expected_stdout")
        missing = [key for key in required if key not in value]
        if missing:
            raise TaskFormatError(f"case missing fields: {missing}")
        if not all(isinstance(value[key], str) for key in required):
            raise TaskFormatError("case_id, stdin, and expected_stdout must be strings")
        if not value["case_id"].strip() or not value["expected_stdout"].strip():
            raise TaskFormatError("case_id and expected_stdout cannot be empty")
        return cls(**{key: value[key] for key in required})


@dataclass(frozen=True)
class CobolTask:
    task_id: str
    group_id: str
    title: str
    specification: str
    program: str
    reference_program: str
    reference_tests: tuple[CobolCase, ...]
    mutants: tuple[dict[str, str], ...]
    source_url: str
    license_id: str
    review_status: ReviewStatus
    review_notes: str = ""

    def __post_init__(self) -> None:
        required_text = (
            "task_id",
            "group_id",
            "title",
            "specification",
            "program",
            "reference_program",
            "source_url",
            "license_id",
        )
        for field_name in required_text:
            if not getattr(self, field_name).strip():
                raise TaskFormatError(f"{field_name} must be non-empty")
        if not self.reference_tests:
            raise TaskFormatError("at least one reference test is required")
        case_ids = [case.case_id for case in self.reference_tests]
        if len(case_ids) != len(set(case_ids)):
            raise TaskFormatError("reference test case IDs must be unique")
        for mutant in self.mutants:
            if not mutant.get("mutant_id") or not mutant.get("source"):
                raise TaskFormatError("each mutant requires mutant_id and source")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CobolTask":
        required = {
            "task_id", "group_id", "title", "specification", "program",
            "reference_program", "reference_tests", "mutants", "source_url",
            "license_id", "review_status",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise TaskFormatError(f"task missing required fields: {missing}")
        try:
            return cls(
                task_id=value["task_id"],
                group_id=value["group_id"],
                title=value["title"],
                specification=value["specification"],
                program=value["program"],
                reference_program=value["reference_program"],
                reference_tests=tuple(
                    CobolCase.from_dict(case) for case in value["reference_tests"]
                ),
                mutants=tuple(value["mutants"]),
                source_url=value["source_url"],
                license_id=value["license_id"],
                review_status=ReviewStatus(value["review_status"]),
                review_notes=value.get("review_notes", ""),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TaskFormatError):
                raise
            raise TaskFormatError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["review_status"] = self.review_status.value
        result["reference_tests"] = [asdict(case) for case in self.reference_tests]
        return result


def dumps_task(task: CobolTask) -> str:
    return json.dumps(task.to_dict(), sort_keys=True, ensure_ascii=False)