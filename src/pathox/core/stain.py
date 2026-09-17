from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StainNormalizationResult:
    image: Image.Image
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]


class MacenkoNormalizer:
    """Macenko-style H&E stain normalization."""

    TARGET_STAIN_MATRIX = np.array(
        [
            [0.650, 0.700, 0.290],
            [0.072, 0.990, 0.105],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        beta: float = 0.15,
        alpha: float = 1.0,
        reference_max_concentration: tuple[float, float] = (1.9705, 1.0308),
    ) -> None:
        if beta <= 0:
            raise ValueError("beta must be positive")

        if alpha <= 0 or alpha >= 50:
            raise ValueError("alpha must be in (0, 50)")

        if (
            reference_max_concentration[0] <= 0
            or reference_max_concentration[1] <= 0
        ):
            raise ValueError(
                "reference_max_concentration must contain positive values"
            )

        self.beta = beta
        self.alpha = alpha
        self.reference_max_concentration = np.asarray(
            reference_max_concentration,
            dtype=np.float64,
        )

    def normalize(self, image: Image.Image) -> StainNormalizationResult:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

        normalized = self._normalize_array(rgb)

        result = Image.fromarray(normalized, mode="RGB")

        return StainNormalizationResult(
            image=result,
            input_shape=image.size,
            output_shape=result.size,
        )

    def _normalize_array(self, rgb: np.ndarray) -> np.ndarray:
        image = rgb.astype(np.float64) + 1.0

        optical_density = -np.log(image / 256.0)

        tissue_mask = np.any(
            optical_density > self.beta,
            axis=2,
        )

        pixels = optical_density[tissue_mask]

        if pixels.shape[0] < 10:
            return rgb.copy()

        covariance = np.cov(pixels, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]

        plane = eigenvectors[:, :2]

        projected = pixels @ plane

        angles = np.arctan2(
            projected[:, 1],
            projected[:, 0],
        )

        min_angle = np.percentile(
            angles,
            self.alpha,
        )

        max_angle = np.percentile(
            angles,
            100.0 - self.alpha,
        )

        vector_min = plane @ np.array(
            [
                np.cos(min_angle),
                np.sin(min_angle),
            ]
        )

        vector_max = plane @ np.array(
            [
                np.cos(max_angle),
                np.sin(max_angle),
            ]
        )

        if vector_min[0] < vector_max[0]:
            stain_matrix = np.column_stack(
                [vector_max, vector_min]
            )
        else:
            stain_matrix = np.column_stack(
                [vector_min, vector_max]
            )

        stain_matrix = np.maximum(stain_matrix, 1e-8)

        stain_matrix /= np.linalg.norm(
            stain_matrix,
            axis=0,
            keepdims=True,
        )

        concentration = pixels @ np.linalg.pinv(
            stain_matrix
        )

        max_concentration = np.percentile(
            concentration,
            99,
            axis=0,
        )

        max_concentration = np.maximum(
            max_concentration,
            1e-8,
        )

        scale = (
            self.reference_max_concentration
            / max_concentration
        )

        normalized_pixels = (
            concentration * scale
        )

        normalized_od = (
            normalized_pixels
            @ self.TARGET_STAIN_MATRIX
        )

        normalized_rgb = (
            256.0 * np.exp(-normalized_od) - 1.0
        )

        output = rgb.copy()

        output[tissue_mask] = np.clip(
            normalized_rgb * 255.0 / 255.0,
            0,
            255,
        ).astype(np.uint8)

        return output


class StainAugmenter:
    """Simple H&E-like color perturbations for domain robustness."""

    def __init__(
        self,
        brightness: float = 0.10,
        contrast: float = 0.10,
        saturation: float = 0.10,
    ) -> None:
        if brightness < 0 or contrast < 0 or saturation < 0:
            raise ValueError("augmentation strengths must be non-negative")

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def apply(
        self,
        image: Image.Image,
        rng: np.random.Generator | None = None,
    ) -> Image.Image:
        rng = rng or np.random.default_rng()

        array = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        )

        brightness_factor = rng.uniform(
            1.0 - self.brightness,
            1.0 + self.brightness,
        )

        contrast_factor = rng.uniform(
            1.0 - self.contrast,
            1.0 + self.contrast,
        )

        array = array * brightness_factor

        mean = array.mean(axis=(0, 1), keepdims=True)

        array = (
            (array - mean) * contrast_factor
            + mean
        )

        gray = (
            0.299 * array[:, :, 0]
            + 0.587 * array[:, :, 1]
            + 0.114 * array[:, :, 2]
        )

        saturation_factor = rng.uniform(
            1.0 - self.saturation,
            1.0 + self.saturation,
        )

        array = (
            gray[:, :, None]
            + (array - gray[:, :, None]) * saturation_factor
        )

        return Image.fromarray(
            np.clip(array, 0, 255).astype(np.uint8),
            mode="RGB",
        )
