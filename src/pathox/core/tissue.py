from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TissueDetectionResult:
    mask: np.ndarray
    tissue_fraction: float


class TissueDetector:
    """Detect tissue regions in an RGB pathology image."""

    def __init__(
        self,
        saturation_threshold: int = 20,
        value_threshold: int = 235,
        kernel_size: int = 5,
        min_component_area: int = 64,
    ) -> None:
        if not 0 <= saturation_threshold <= 255:
            raise ValueError("saturation_threshold must be in [0, 255]")

        if not 0 <= value_threshold <= 255:
            raise ValueError("value_threshold must be in [0, 255]")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")

        if min_component_area < 0:
            raise ValueError("min_component_area must be non-negative")

        self.saturation_threshold = saturation_threshold
        self.value_threshold = value_threshold
        self.kernel_size = kernel_size
        self.min_component_area = min_component_area

    def detect(self, image: Image.Image) -> TissueDetectionResult:
        rgb = np.asarray(image.convert("RGB"))

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        mask = (
            (saturation > self.saturation_threshold)
            & (value < self.value_threshold)
        ).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.kernel_size, self.kernel_size),
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if self.min_component_area > 0:
            mask = self._remove_small_components(
                mask,
                self.min_component_area,
            )

        binary = mask > 0
        tissue_fraction = float(binary.mean())

        return TissueDetectionResult(
            mask=mask,
            tissue_fraction=tissue_fraction,
        )

    @staticmethod
    def _remove_small_components(
        mask: np.ndarray,
        min_area: int,
    ) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        output = np.zeros_like(mask)

        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]

            if area >= min_area:
                output[labels == label] = 255

        return output
