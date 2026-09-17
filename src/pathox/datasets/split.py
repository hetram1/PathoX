from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panda import PANDARecord


@dataclass(frozen=True)
class DatasetSplit:
    train: list[PANDARecord]
    validation: list[PANDARecord]


def split_records(
    records: list[PANDARecord],
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> DatasetSplit:
    """Split slides before tile extraction."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    if len(records) < 2:
        raise ValueError(
            "At least two records are required"
        )

    grouped: dict[str, list[PANDARecord]] = {}

    for record in records:
        group_id = record.image_id
        grouped.setdefault(group_id, []).append(record)

    group_ids = list(grouped)

    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)

    validation_count = max(
        1,
        round(len(group_ids) * validation_fraction),
    )

    if validation_count >= len(group_ids):
        validation_count = len(group_ids) - 1

    validation_groups = set(
        group_ids[:validation_count]
    )

    train: list[PANDARecord] = []
    validation: list[PANDARecord] = []

    for group_id, group_records in grouped.items():
        if group_id in validation_groups:
            validation.extend(group_records)
        else:
            train.extend(group_records)

    train.sort(key=lambda record: record.image_id)
    validation.sort(key=lambda record: record.image_id)

    return DatasetSplit(
        train=train,
        validation=validation,
    )
