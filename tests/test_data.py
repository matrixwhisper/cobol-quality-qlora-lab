"""Schema, provenance, split, and curation regression tests."""

from __future__ import annotations

import unittest

from data_quality.curation import audit_tasks
from data_quality.schema import CobolCase, CobolTask, ReviewStatus
from data_quality.splits import make_splits


def make_task(task_id: str, group_id: str | None = None) -> CobolTask:
    return CobolTask(
        task_id=task_id,
        group_id=group_id or task_id,
        title=f"Sum task {task_id}",
        specification="Read signed integers and display their total.",
        program="PROGRAM SOURCE",
        reference_program="REFERENCE SOURCE",
        reference_tests=(CobolCase("basic", "1\n2\n", "SUM=3"),),
        mutants=(),
        source_url="https://example.org/source",
        license_id="CC-BY-4.0",
        review_status=ReviewStatus.ACCEPTED,
        review_notes="Reviewed against the stated behavior and tests.",
    )


class SchemaTests(unittest.TestCase):
    def test_task_json_round_trip(self) -> None:
        task = make_task("sum-1")
        self.assertEqual(CobolTask.from_dict(task.to_dict()), task)

    def test_missing_provenance_is_rejected(self) -> None:
        value = make_task("sum-1").to_dict()
        value["license_id"] = ""
        with self.assertRaises(ValueError):
            CobolTask.from_dict(value)

    def test_bad_case_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CobolCase.from_dict({
                "case_id": "",
                "stdin": "",
                "expected_stdout": "",
            })


class CurationAndSplitTests(unittest.TestCase):
    def test_audit_counts_review_state(self) -> None:
        report = audit_tasks([
            make_task("accepted"),
            CobolTask(
                **{
                    **make_task("unreviewed").__dict__,
                    "review_status": ReviewStatus.UNREVIEWED,
                    "review_notes": "",
                }
            ),
        ])
        self.assertEqual(report.task_count, 2)
        self.assertEqual(report.accepted_count, 1)
        self.assertEqual(report.unreviewed_count, 1)

    def test_same_group_never_crosses_splits(self) -> None:
        tasks = [
            make_task("a", "group-a"),
            make_task("b", "group-a"),
            make_task("c", "group-c"),
            make_task("d", "group-d"),
            make_task("e", "group-e"),
            make_task("f", "group-f"),
        ]
        splits = make_splits(tasks, seed=7)
        owner = {
            task.task_id: split_name
            for split_name in ("train", "validation", "test")
            for task in splits.get(split_name)
        }
        self.assertEqual(owner["a"], owner["b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)