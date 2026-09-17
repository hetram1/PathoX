from pathlib import Path

from pathox.datasets import PANDARecord, split_records


def make_records(count: int) -> list[PANDARecord]:
    return [
        PANDARecord(
            image_id=f"slide_{index}",
            data_provider="radboud",
            isup_grade=index % 6,
            gleason_score="3+4",
            image_path=Path(f"{index}.tiff"),
            mask_path=Path(f"{index}.tiff"),
        )
        for index in range(count)
    ]


def test_split_is_deterministic() -> None:
    records = make_records(10)

    first = split_records(
        records,
        validation_fraction=0.2,
        seed=42,
    )

    second = split_records(
        records,
        validation_fraction=0.2,
        seed=42,
    )

    assert [
        record.image_id
        for record in first.train
    ] == [
        record.image_id
        for record in second.train
    ]

    assert [
        record.image_id
        for record in first.validation
    ] == [
        record.image_id
        for record in second.validation
    ]


def test_train_and_validation_do_not_overlap() -> None:
    records = make_records(10)

    split = split_records(
        records,
        validation_fraction=0.2,
        seed=42,
    )

    train_ids = {
        record.image_id
        for record in split.train
    }

    validation_ids = {
        record.image_id
        for record in split.validation
    }

    assert train_ids.isdisjoint(validation_ids)
    assert len(train_ids | validation_ids) == 10


def test_split_preserves_all_records() -> None:
    records = make_records(13)

    split = split_records(
        records,
        validation_fraction=0.2,
        seed=123,
    )

    assert (
        len(split.train)
        + len(split.validation)
        == len(records)
    )


def test_invalid_fraction() -> None:
    records = make_records(4)

    try:
        split_records(
            records,
            validation_fraction=1.0,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError"
    )
