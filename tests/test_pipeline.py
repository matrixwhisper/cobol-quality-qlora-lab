"""Mutation-score tests using real GnuCOBOL compilation when available."""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from benchmark.pipeline import (
    evaluate_programs,
    score_generated_suite,
    validate_generated_suite,
)
from data_quality.schema import CobolCase, CobolTask, ReviewStatus


ROOT = Path(__file__).resolve().parents[1]
BUGGY = (ROOT / "benchmark" / "buggy_sum.cob").read_text(encoding="utf-8")
REFERENCE = (ROOT / "benchmark" / "reference_sum.cob").read_text(encoding="utf-8")


def make_task() -> CobolTask:
    return CobolTask(
        task_id="sum-001",
        group_id="sum-group",
        title="Sum a sequence of signed integers",
        specification="Read a count and then that many signed integers; print their sum.",
        program=BUGGY,
        reference_program=REFERENCE,
        reference_tests=(
            CobolCase("positive", "2\n3\n4\n", "SUM=7"),
            CobolCase("negative", "2\n5\n-3\n", "SUM=2"),
            CobolCase("empty", "0\n", "SUM=0"),
        ),
        mutants=({"mutant_id": "ignores-negative", "source": BUGGY},),
        source_url="https://example.org/original-fixture",
        license_id="CC0-1.0",
        review_status=ReviewStatus.ACCEPTED,
        review_notes="Original project fixture; output checked using GnuCOBOL.",
    )


class GeneratedSuiteTests(unittest.TestCase):
    def test_valid_suite_parses(self) -> None:
        cases = validate_generated_suite({
            "cases": [{"case_id": "empty", "stdin": "0\n", "expected_stdout": "SUM=0"}]
        })
        self.assertEqual(len(cases), 1)

    def test_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_generated_suite({
                "cases": [
                    {"case_id": "x", "stdin": "0\n", "expected_stdout": "SUM=0"},
                    {"case_id": "x", "stdin": "1\n1\n", "expected_stdout": "SUM=1"},
                ]
            })

    def test_malformed_model_output_is_a_parse_failure(self) -> None:
        result = score_generated_suite(make_task(), "not JSON")
        self.assertIn("parse_error", result)
        self.assertIsNone(result["mutation_score"])


@unittest.skipUnless(
    shutil.which("cobc"),
    "GnuCOBOL is not installed",
)
class GnuCobolTests(unittest.TestCase):
    def test_generated_cases_kill_known_mutant(self) -> None:
        task = make_task()
        suite = (
            CobolCase("positive", "2\n3\n4\n", "SUM=7"),
            CobolCase("negative", "2\n5\n-3\n", "SUM=2"),
        )
        result = evaluate_programs(task, suite)
        self.assertEqual(result["oracle_valid_count"], 2)
        self.assertEqual(result["mutants_killed"], 1)
        self.assertEqual(result["mutation_score"], 1.0)

    def test_bad_expected_output_is_not_counted_as_a_kill(self) -> None:
        task = make_task()
        suite = (CobolCase("bad-oracle", "1\n4\n", "SUM=999"),)
        result = evaluate_programs(task, suite)
        self.assertEqual(result["oracle_invalid_count"], 1)
        self.assertEqual(result["mutants_killed"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)