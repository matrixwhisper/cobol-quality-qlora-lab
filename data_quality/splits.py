"""Deterministic group-aware splitting with leakage checks."""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from typing import Literal

from data_quality.schema import CobolTask


SplitName = Literal["train", "validation", "test"]


class SplitError(ValueError):
    """Raised when valid leakage-safe splits cannot be constructed."""


@dataclass(frozen=True)
class TaskSplits:
    train: tuple[CobolTask, ...]
    validation: tuple[CobolTask, ...]
    test: tuple[CobolTask, ...]
    seed: int

    def get(self, name: str) -> tuple[CobolTask, ...]:
        if name not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {name!r}")
        return getattr(self, name)

    def manifest(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "task_ids": {
                name: [task.task_id for task in self.get(name)]
                for name in ("train", "validation", "test")
            },
            "group_ids": {
                name: sorted({task.group_id for task in self.get(name)})
                for name in ("train", "validation", "test")
            },
        }


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def make_splits(
    tasks: list[CobolTask],
    seed: int = 2026,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TaskSplits:
    if not tasks:
        raise SplitError("cannot split an empty dataset")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise SplitError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise SplitError("fractions must leave test examples")

    groups: dict[str, list[CobolTask]] = {}
    task_ids: set[str] = set()
    for task in tasks:
        if task.task_id in task_ids:
            raise SplitError(f"duplicate task ID: {task.task_id}")
        task_ids.add(task.task_id)
        groups.setdefault(task.group_id, []).append(task)

    if len(groups) < 3:
        raise SplitError("need at least three independent task groups")

    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    target_train = max(1, int(len(tasks) * train_fraction))
    target_validation = max(1, int(len(tasks) * validation_fraction))
    partitions: dict[str, list[CobolTask]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    # Seed every partition with a distinct group, then greedily approach targets.
    initial = {"train": group_ids[0], "validation": group_ids[1], "test": group_ids[2]}
    for split_name, group_id in initial.items():
        partitions[split_name].extend(groups[group_id])
    remaining = group_ids[3:]

    for group_id in remaining:
        group = groups[group_id]
        train_deficit = target_train - len(partitions["train"])
        validation_deficit = target_validation - len(partitions["validation"])
        if train_deficit >= validation_deficit and train_deficit > 0:
            destination = "train"
        elif validation_deficit > 0:
            destination = "validation"
        else:
            destination = "test"
        partitions[destination].extend(group)

    result = TaskSplits(
        train=tuple(partitions["train"]),
        validation=tuple(partitions["validation"]),
        test=tuple(partitions["test"]),
        seed=seed,
    )
    assert_no_leakage(result)
    return result


def assert_no_leakage(splits: TaskSplits) -> None:
    owner: dict[str, str] = {}
    prompt_owner: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for task in splits.get(split_name):
            prior = owner.setdefault(task.group_id, split_name)
            if prior != split_name:
                raise SplitError(f"group {task.group_id!r} crosses splits")
            prompt = _normalize(task.specification)
            prior_prompt = prompt_owner.setdefault(prompt, split_name)
            if prior_prompt != split_name:
                raise SplitError("duplicate normalized specification crosses splits")