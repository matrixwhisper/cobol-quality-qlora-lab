"""Compile COBOL and score generated test suites against reference/mutants."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data_quality.schema import CobolCase, CobolTask


class CobolToolchainError(RuntimeError):
    """GnuCOBOL is missing or compilation failed."""


def normalize_output(value: str) -> str:
    return "\n".join(
        re.sub(r"^(SUM=)\s+", r"\1", line.strip())
        for line in value.replace("\r", "").split("\n")
        if line.strip()
    )


def validate_generated_suite(value: Any) -> tuple[CobolCase, ...]:
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("generated suite must be a JSON object with a cases list")
    cases = tuple(CobolCase.from_dict(case) for case in value["cases"])
    if not cases:
        raise ValueError("generated suite must include at least one test case")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("generated case IDs must be unique")
    return cases


def _compile(source: str, workdir: Path, name: str, compiler: str) -> Path:
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise CobolToolchainError(
            "GnuCOBOL not found. Install the cobc compiler before evaluation."
        )
    source_path = workdir / f"{name}.cob"
    executable = workdir / name
    source_path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            compiler_path, "-x", "-free", "-Wall", "-Wextra",
            "-o", str(executable), str(source_path),
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        raise CobolToolchainError(
            f"COBOL compilation failed for {name}:\n{result.stderr}"
        )
    return executable


def _run(executable: Path, case: CobolCase, timeout: float) -> dict[str, object]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [str(executable)],
            input=case.stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "return_code": None,
            "stdout": "",
            "stderr": "timeout",
            "timed_out": True,
            "duration_seconds": time.perf_counter() - started,
        }
    return {
        "return_code": result.returncode,
        "stdout": normalize_output(result.stdout),
        "stderr": result.stderr[:4000],
        "timed_out": False,
        "duration_seconds": time.perf_counter() - started,
    }


def evaluate_programs(
    task: CobolTask,
    generated_cases: tuple[CobolCase, ...],
    timeout_seconds: float = 2.0,
    compiler: str = "cobc",
) -> dict[str, object]:
    if timeout_seconds <= 0 or timeout_seconds > 30:
        raise ValueError("timeout_seconds must be in (0, 30]")

    with tempfile.TemporaryDirectory(prefix="cobol_benchmark_") as temp:
        root = Path(temp)
        ref_dir = root / "reference"
        ref_dir.mkdir()
        ref_exe = _compile(task.reference_program, ref_dir, "reference", compiler)

        mutant_executables = []
        for index, mutant in enumerate(task.mutants):
            mutant_dir = root / f"mutant_{index}"
            mutant_dir.mkdir()
            mutant_exe = _compile(
                mutant["source"], mutant_dir, mutant["mutant_id"], compiler
            )
            mutant_executables.append((mutant["mutant_id"], mutant_exe))

        oracle_valid = []
        for case in generated_cases:
            reference_result = _run(ref_exe, case, timeout_seconds)
            expected = normalize_output(case.expected_stdout)
            valid = (
                not reference_result["timed_out"]
                and reference_result["return_code"] == 0
                and reference_result["stdout"] == expected
            )
            oracle_valid.append(valid)

        per_mutant = []
        for mutant_id, executable in mutant_executables:
            killed_cases = []
            for case, valid in zip(generated_cases, oracle_valid):
                if not valid:
                    continue
                result = _run(executable, case, timeout_seconds)
                killed = (
                    result["timed_out"]
                    or result["return_code"] != 0
                    or result["stdout"] != normalize_output(case.expected_stdout)
                )
                if killed:
                    killed_cases.append(case.case_id)
            per_mutant.append({
                "mutant_id": mutant_id,
                "killed": bool(killed_cases),
                "killed_by_cases": killed_cases,
            })

    eligible_mutants = len(per_mutant)
    killed_count = sum(item["killed"] for item in per_mutant)
    return {
        "task_id": task.task_id,
        "generated_case_count": len(generated_cases),
        "oracle_valid_count": sum(oracle_valid),
        "oracle_invalid_count": len(oracle_valid) - sum(oracle_valid),
        "mutant_count": eligible_mutants,
        "mutants_killed": killed_count,
        "mutation_score": killed_count / eligible_mutants if eligible_mutants else None,
        "mutants": per_mutant,
    }


def score_generated_suite(
    task: CobolTask,
    model_output: str,
    timeout_seconds: float = 2.0,
) -> dict[str, object]:
    try:
        start = model_output.find("{")
        end = model_output.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("model output did not contain a JSON object")
        parsed = json.loads(model_output[start:end])
        cases = validate_generated_suite(parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "task_id": task.task_id,
            "parse_error": str(exc),
            "oracle_valid_count": 0,
            "mutation_score": None,
        }
    result = evaluate_programs(task, cases, timeout_seconds)
    result["parse_error"] = None
    return result


def write_report(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target