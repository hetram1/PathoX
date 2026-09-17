from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openslide
from PIL import Image


@dataclass(frozen=True)
class WSIMetadata:
    path: str
    vendor: str | None
    level_count: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    mpp_x: float | None
    mpp_y: float | None


class WSIReader:
    """Safe wrapper around OpenSlide for whole-slide image access."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"WSI file not found: {self.path}")

        if not self.path.is_file():
            raise ValueError(f"WSI path is not a file: {self.path}")

        try:
            self.slide = openslide.OpenSlide(str(self.path))
        except openslide.OpenSlideUnsupportedFormatError as exc:
            raise ValueError(
                f"Unsupported whole-slide image format: {self.path}"
            ) from exc

    @property
    def metadata(self) -> WSIMetadata:
        properties = self.slide.properties

        mpp_x = self._get_float_property(
            properties,
            openslide.PROPERTY_NAME_MPP_X,
        )

        mpp_y = self._get_float_property(
            properties,
            openslide.PROPERTY_NAME_MPP_Y,
        )

        vendor = properties.get(openslide.PROPERTY_NAME_VENDOR)

        return WSIMetadata(
            path=str(self.path),
            vendor=vendor,
            level_count=self.slide.level_count,
            level_dimensions=tuple(self.slide.level_dimensions),
            level_downsamples=tuple(self.slide.level_downsamples),
            mpp_x=mpp_x,
            mpp_y=mpp_y,
        )

    def read_region(
        self,
        x: int,
        y: int,
        level: int,
        width: int,
        height: int,
    ) -> Image.Image:
        """Read an RGB tile.

        x and y are always specified in level-0 coordinates.
        """
        if level < 0 or level >= self.slide.level_count:
            raise ValueError(f"Invalid pyramid level: {level}")

        if width <= 0 or height <= 0:
            raise ValueError("Tile width and height must be positive")

        if x < 0 or y < 0:
            raise ValueError("Tile coordinates must be non-negative")

        return self.slide.read_region(
            (x, y),
            level,
            (width, height),
        ).convert("RGB")

    def thumbnail(self, width: int, height: int | None = None) -> Image.Image:
        """Generate a thumbnail without loading the full-resolution slide."""
        if width <= 0:
            raise ValueError("Thumbnail width must be positive")

        if height is None:
            height = width

        if height <= 0:
            raise ValueError("Thumbnail height must be positive")

        return self.slide.get_thumbnail((width, height)).convert("RGB")

    def close(self) -> None:
        """Release the OpenSlide handle."""
        self.slide.close()

    def __enter__(self) -> "WSIReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @staticmethod
    def _get_float_property(
        properties: openslide.OpenSlideProperties,
        key: str,
    ) -> float | None:
        value = properties.get(key)

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None
