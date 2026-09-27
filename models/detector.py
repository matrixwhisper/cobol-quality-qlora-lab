"""Random-forest detector trained against human review outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from data_quality.schema import CobolTask, ReviewStatus
from models.scoring import matrix


class QualityDetector:
    def __init__(self) -> None:
        self.classifier = None
        self.feature_names: list[str] = []

    def fit(self, tasks: Iterable[CobolTask]) -> "QualityDetector":
        from sklearn.ensemble import RandomForestClassifier

        labeled = [
            task for task in tasks
            if task.review_status is not ReviewStatus.UNREVIEWED
        ]
        if len(labeled) < 8:
            raise ValueError("at least 8 human-reviewed tasks are required")
        labels = [
            int(task.review_status is not ReviewStatus.ACCEPTED)
            for task in labeled
        ]
        if len(set(labels)) != 2:
            raise ValueError("training labels must include accepted and revise/reject tasks")

        self.feature_names, values = matrix(labeled)
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=2026,
            n_jobs=-1,
        )
        self.classifier.fit(values, labels)
        return self

    def predict_review_risk(self, tasks: Iterable[CobolTask]) -> list[dict[str, object]]:
        if self.classifier is None:
            raise RuntimeError("detector has not been trained")
        rows = list(tasks)
        _, values = matrix(rows)
        probabilities = self.classifier.predict_proba(values)[:, 1]
        return [
            {"task_id": task.task_id, "review_risk": float(score)}
            for task, score in zip(rows, probabilities)
        ]

    def save(self, path: str | Path) -> Path:
        if self.classifier is None:
            raise RuntimeError("cannot save an unfitted detector")
        import joblib

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"classifier": self.classifier, "feature_names": self.feature_names},
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "QualityDetector":
        import joblib

        saved = joblib.load(path)
        detector = cls()
        detector.classifier = saved["classifier"]
        detector.feature_names = saved["feature_names"]
        return detector