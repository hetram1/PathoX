from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PANDASegmentationDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split=None,
        image_size=512,
        augment=False,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.augment = augment

        with self.manifest_path.open(newline="") as f:
            records = list(csv.DictReader(f))

        if split is not None:
            records = [r for r in records if r["split"] == split]

        if not records:
            raise ValueError(f"No records found for split={split}")

        self.records = records

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _load_image(path):
        image = Image.open(path).convert("RGB")
        return np.array(image, dtype=np.uint8, copy=True)

    @staticmethod
    def _load_mask(path):
        mask = Image.open(path).convert("L")
        return np.array(mask, dtype=np.uint8, copy=True)

    def __getitem__(self, idx):
        record = self.records[idx]

        image = self._load_image(record["image_path"])
        mask = self._load_mask(record["mask_path"])

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask mismatch: {image.shape[:2]} vs {mask.shape}"
            )

        if image.shape[:2] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size}, "
                f"got {image.shape[:2]}"
            )

        labels = np.unique(mask)
        if not set(labels.tolist()).issubset({0, 1, 2, 3, 4, 5}):
            raise ValueError(
                f"Unexpected mask labels: {labels.tolist()}"
            )

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            if np.random.rand() < 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        image = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask
