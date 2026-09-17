from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RADBOUD_MASK_CLASSES = {
    0: "background",
    1: "stroma",
    2: "benign_epithelium",
    3: "gleason_3",
    4: "gleason_4",
    5: "gleason_5",
}

KAROLINSKA_MASK_CLASSES = {
    0: "background",
    1: "benign_tissue",
    2: "cancerous_tissue",
}


@dataclass(frozen=True)
class PANDARecord:
    image_id: str
    data_provider: str
    isup_grade: int
    gleason_score: str
    image_path: Path | None
    mask_path: Path | None

    @property
    def has_mask(self) -> bool:
        return self.mask_path is not None

    @property
    def supports_six_class_segmentation(self) -> bool:
        return (
            self.data_provider.lower() == "radboud"
            and self.has_mask
        )


def _find_slide(
    root: Path,
    image_id: str,
) -> Path | None:
    if not root.exists():
        return None

    extensions = (
        ".tiff",
        ".tif",
        ".svs",
    )

    for extension in extensions:
        candidate = root / f"{image_id}{extension}"

        if candidate.is_file():
            return candidate

    return None


def _find_mask(
    root: Path,
    image_id: str,
) -> Path | None:
    if not root.exists():
        return None

    candidates = (
        root / f"{image_id}.tiff",
        root / f"{image_id}.tif",
        root / f"{image_id}_mask.tiff",
        root / f"{image_id}_mask.tif",
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def load_panda_manifest(
    csv_path: str | Path,
    image_root: str | Path | None = None,
    mask_root: str | Path | None = None,
) -> list[PANDARecord]:
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"PANDA CSV not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    required_columns = {
        "image_id",
        "data_provider",
        "isup_grade",
        "gleason_score",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Missing PANDA columns: "
            + ", ".join(sorted(missing))
        )

    image_root_path = (
        Path(image_root)
        if image_root is not None
        else None
    )

    mask_root_path = (
        Path(mask_root)
        if mask_root is not None
        else None
    )

    records: list[PANDARecord] = []

    for row in dataframe.itertuples(
        index=False
    ):
        image_id = str(row.image_id)

        image_path = (
            _find_slide(
                image_root_path,
                image_id,
            )
            if image_root_path is not None
            else None
        )

        mask_path = (
            _find_mask(
                mask_root_path,
                image_id,
            )
            if mask_root_path is not None
            else None
        )

        records.append(
            PANDARecord(
                image_id=image_id,
                data_provider=str(
                    row.data_provider
                ),
                isup_grade=int(
                    row.isup_grade
                ),
                gleason_score=str(
                    row.gleason_score
                ),
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    return records


def segmentation_records(
    records: list[PANDARecord],
) -> list[PANDARecord]:
    return [
        record
        for record in records
        if record.supports_six_class_segmentation
    ]
