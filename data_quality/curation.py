"""Conservative dataset audits; findings require human adjudication."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from data_quality.schema import CobolTask, ReviewStatus


VAGUE_TEXT = re.compile(r"\b(?:etc\.?|as appropriate|some cases|and so on)\b", re.I)


@dataclass(frozen=True)
class CurationIssue:
    task_id: str
    code: str
    severity: str
    detail: str


@dataclass(frozen=True)
class CurationReport:
    task_count: int
    accepted_count: int
    revised_count: int
    rejected_count: int
    unreviewed_count: int
    issues_by_code: dict[str, int]
    issues: tuple[CurationIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_count": self.task_count,
            "accepted_count": self.accepted_count,
            "revised_count": self.revised_count,
            "rejected_count": self.rejected_count,
            "unreviewed_count": self.unreviewed_count,
            "issues_by_code": self.issues_by_code,
            "issues": [asdict(issue) for issue in self.issues],
        }


def source_fingerprint(tasks: Iterable[CobolTask]) -> str:
    digest = hashlib.sha256()
    for task in sorted(tasks, key=lambda item: item.task_id):
        digest.update(task.task_id.encode())
        digest.update(task.program.encode())
        digest.update(task.reference_program.encode())
        digest.update(task.specification.encode())
    return digest.hexdigest()


def audit_tasks(tasks: Iterable[CobolTask]) -> CurationReport:
    records = list(tasks)
    issues: list[CurationIssue] = []
    seen_programs: dict[str, str] = {}
    status_counts = Counter(task.review_status.value for task in records)

    for task in records:
        normalized = re.sub(r"\s+", " ", task.program).strip().casefold()
        program_hash = hashlib.sha256(normalized.encode()).hexdigest()
        previous = seen_programs.get(program_hash)
        if previous is not None:
            issues.append(CurationIssue(
                task.task_id, "duplicate_program", "review",
                f"program duplicates task {previous}",
            ))
        else:
            seen_programs[program_hash] = task.task_id

        if VAGUE_TEXT.search(task.specification):
            issues.append(CurationIssue(
                task.task_id, "possible_ambiguity", "review",
                "specification contains a configured vague phrase",
            ))
        if not task.source_url.startswith(("https://", "http://")):
            issues.append(CurationIssue(
                task.task_id, "missing_provenance_url", "error",
                "source_url must identify a reviewable source",
            ))
        if task.license_id.casefold() in {"unknown", "unspecified", "n/a"}:
            issues.append(CurationIssue(
                task.task_id, "unknown_license", "error",
                "license must be identified before redistribution or training",
            ))
        if task.review_status is ReviewStatus.ACCEPTED and not task.review_notes.strip():
            issues.append(CurationIssue(
                task.task_id, "missing_review_rationale", "warning",
                "accepted tasks should record why they passed review",
            ))

    return CurationReport(
        task_count=len(records),
        accepted_count=status_counts[ReviewStatus.ACCEPTED.value],
        revised_count=status_counts[ReviewStatus.REVISED.value],
        rejected_count=status_counts[ReviewStatus.REJECTED.value],
        unreviewed_count=status_counts[ReviewStatus.UNREVIEWED.value],
        issues_by_code=dict(sorted(Counter(issue.code for issue in issues).items())),
        issues=tuple(issues),
    )