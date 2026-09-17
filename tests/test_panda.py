from pathlib import Path

import pandas as pd
import pytest

from pathox.datasets import (
    RADBOUD_MASK_CLASSES,
    load_panda_manifest,
    segmentation_records,
)


def test_radboud_has_six_classes() -> None:
    assert len(RADBOUD_MASK_CLASSES) == 6
    assert RADBOUD_MASK_CLASSES[3] == "gleason_3"
    assert RADBOUD_MASK_CLASSES[5] == "gleason_5"


def test_manifest_loading(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"

    image_root.mkdir()
    mask_root.mkdir()

    image_path = image_root / "abc.tiff"
    mask_path = mask_root / "abc.tiff"

    image_path.write_bytes(b"image")
    mask_path.write_bytes(b"mask")

    dataframe = pd.DataFrame(
        [
            {
                "image_id": "abc",
                "data_provider": "radboud",
                "isup_grade": 3,
                "gleason_score": "4+3",
            },
            {
                "image_id": "xyz",
                "data_provider": "karolinska",
                "isup_grade": 2,
                "gleason_score": "3+4",
            },
        ]
    )

    csv_path = tmp_path / "train.csv"
    dataframe.to_csv(
        csv_path,
        index=False,
    )

    records = load_panda_manifest(
        csv_path,
        image_root=image_root,
        mask_root=mask_root,
    )

    assert len(records) == 2
    assert records[0].image_path == image_path
    assert records[0].mask_path == mask_path
    assert records[0].supports_six_class_segmentation

    assert not records[1].supports_six_class_segmentation


def test_missing_columns_are_rejected(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "bad.csv"

    pd.DataFrame(
        [
            {
                "image_id": "abc",
            }
        ]
    ).to_csv(
        csv_path,
        index=False,
    )

    with pytest.raises(ValueError):
        load_panda_manifest(csv_path)


def test_segmentation_records() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "image_id": "rad",
                "data_provider": "Radboud",
                "isup_grade": 4,
                "gleason_score": "4+4",
            },
            {
                "image_id": "kar",
                "data_provider": "Karolinska",
                "isup_grade": 2,
                "gleason_score": "3+4",
            },
        ]
    )

    records = []

    for row in dataframe.itertuples(
        index=False
    ):
        from pathox.datasets import PANDARecord

        records.append(
            PANDARecord(
                image_id=row.image_id,
                data_provider=row.data_provider,
                isup_grade=row.isup_grade,
                gleason_score=row.gleason_score,
                image_path=Path("image.tiff"),
                mask_path=Path("mask.tiff"),
            )
        )

    filtered = segmentation_records(records)

    assert len(filtered) == 1
    assert filtered[0].image_id == "rad"
