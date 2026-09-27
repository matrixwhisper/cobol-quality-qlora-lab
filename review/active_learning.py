"""Uncertainty-ranked, group-diverse human-review queues."""

from __future__ import annotations

from data_quality.schema import CobolTask, ReviewStatus
from models.detector import QualityDetector


def build_review_queue(
    detector: QualityDetector,
    tasks: list[CobolTask],
    batch_size: int = 25,
) -> list[dict[str, object]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    candidates = [
        task for task in tasks
        if task.review_status is ReviewStatus.UNREVIEWED
    ]
    predictions = detector.predict_proba(candidates)
    ranked = sorted(
        zip(candidates, predictions),
        key=lambda pair: abs(
            float(pair[1]["probability_review_needed"]) - 0.5
        ),
    )
    chosen = []
    seen_groups: set[str] = set()
    for task, prediction in ranked:
        if task.group_id in seen_groups:
            continue
        chosen.append({
            "task_id": task.task_id,
            "group_id": task.group_id,
            "probability_review_needed": prediction["probability_review_needed"],
            "specification": task.specification,
            "program": task.program,
            "provenance": task.source_url,
        })
        seen_groups.add(task.group_id)
        if len(chosen) >= batch_size:
            break
    return chosen