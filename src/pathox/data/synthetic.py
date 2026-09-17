from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticSegmentationConfig:
    samples: int = 32
    image_size: int = 128
    num_classes: int = 6
    seed: int = 42


class SyntheticSegmentationDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor]]
):
    """Synthetic dataset used only to validate the training pipeline."""

    def __init__(
        self,
        config: SyntheticSegmentationConfig | None = None,
    ) -> None:
        self.config = (
            config
            or SyntheticSegmentationConfig()
        )

        if self.config.samples <= 0:
            raise ValueError("samples must be positive")

        if self.config.image_size <= 0:
            raise ValueError("image_size must be positive")

        if self.config.num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2"
            )

        rng = np.random.default_rng(
            self.config.seed
        )

        self.images: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []

        for _ in range(self.config.samples):
            image, mask = self._create_sample(rng)

            self.images.append(
                torch.from_numpy(image)
            )

            self.masks.append(
                torch.from_numpy(mask)
            )

    def _create_sample(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        size = self.config.image_size

        image = np.zeros(
            (3, size, size),
            dtype=np.float32,
        )

        image += rng.uniform(
            0.05,
            0.15,
            size=image.shape,
        ).astype(np.float32)

        mask = np.zeros(
            (size, size),
            dtype=np.int64,
        )

        for class_id in range(
            1,
            self.config.num_classes,
        ):
            cx = int(
                rng.integers(
                    size // 8,
                    size * 7 // 8,
                )
            )

            cy = int(
                rng.integers(
                    size // 8,
                    size * 7 // 8,
                )
            )

            radius = int(
                rng.integers(
                    max(6, size // 16),
                    max(8, size // 7),
                )
            )

            yy, xx = np.ogrid[
                :size,
                :size,
            ]

            shape = (
                (xx - cx) ** 2
                + (yy - cy) ** 2
                <= radius**2
            )

            mask[shape] = class_id

            intensity = class_id / (
                self.config.num_classes - 1
            )

            image[0, shape] = intensity
            image[1, shape] = 1.0 - intensity
            image[2, shape] = (
                0.25 + 0.5 * intensity
            )

        image = np.clip(
            image,
            0.0,
            1.0,
        )

        return image, mask

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.images[index],
            self.masks[index],
        )
