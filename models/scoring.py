"""Structural feature extraction for COBOL data-quality triage."""

from __future__ import annotations

import re
from collections import Counter

from data_quality.schema import CobolTask


VERBS = {
    "ACCEPT", "ADD", "CALL", "COMPUTE", "DISPLAY", "DIVIDE",
    "EVALUATE", "IF", "MOVE", "MULTIPLY", "PERFORM", "READ",
    "SEARCH", "STOP", "WRITE",
}


def extract_features(task: CobolTask) -> dict[str, float]:
    source = task.program.upper()
    tokens = re.findall(r"[A-Z][A-Z0-9-]*", source)
    counts = Counter(tokens)
    nonempty_lines = [line for line in source.splitlines() if line.strip()]
    return {
        "program_chars": float(len(task.program)),
        "program_lines": float(len(nonempty_lines)),
        "token_count": float(len(tokens)),
        "unique_token_ratio": len(set(tokens)) / max(1, len(tokens)),
        "verb_count": float(sum(counts[verb] for verb in VERBS)),
        "copybook_count": float(len(re.findall(r"\bCOPY\b", source))),
        "section_count": float(len(re.findall(r"\bSECTION\.", source))),
        "paragraph_count": float(len(re.findall(r"(?m)^[A-Z0-9-]+\.$", source))),
        "spec_words": float(len(task.specification.split())),
        "reference_case_count": float(len(task.reference_tests)),
        "mutant_count": float(len(task.mutants)),
        "has_source_url": float(task.source_url.startswith("http")),
        "has_known_license": float(task.license_id.lower() not in {"unknown", "unspecified"}),
    }


def matrix(tasks: list[CobolTask]) -> tuple[list[str], list[list[float]]]:
    if not tasks:
        return [], []
    names = sorted(extract_features(tasks[0]))
    return names, [
        [extract_features(task)[name] for name in names]
        for task in tasks
    ]