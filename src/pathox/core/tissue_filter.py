from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable, Iterator

import numpy as np

from .tile import Tile
from .tissue import TissueDetector
from .wsi import WSIReader


@dataclass(frozen=True)
class TileScore:
    tile: Tile
    tissue_fraction: float
    is_candidate: bool


class TissueTileFilter:
    """Filter WSI tiles using a low-resolution tissue mask."""

    def __init__(
        self,
        reader: WSIReader,
        detector: TissueDetector | None = None,
        thumbnail_width: int = 1024,
        min_tissue_fraction: float = 0.10,
    ) -> None:
        if thumbnail_width <= 0:
            raise ValueError("thumbnail_width must be positive")

        if not 0.0 <= min_tissue_fraction <= 1.0:
            raise ValueError("min_tissue_fraction must be in [0, 1]")

        self.reader = reader
        self.detector = detector or TissueDetector()
        self.thumbnail = reader.thumbnail(thumbnail_width)
        self.result = self.detector.detect(self.thumbnail)

        self.level0_width, self.level0_height = (
            reader.slide.level_dimensions[0]
        )

        self.min_tissue_fraction = min_tissue_fraction

    def _thumbnail_bounds(self, tile: Tile) -> tuple[int, int, int, int]:
        downsample = self.reader.slide.level_downsamples[tile.level]

        level0_x0 = tile.x * downsample
        level0_y0 = tile.y * downsample
        level0_x1 = (tile.x + tile.width) * downsample
        level0_y1 = (tile.y + tile.height) * downsample

        scale_x = self.thumbnail.width / self.level0_width
        scale_y = self.thumbnail.height / self.level0_height

        x0 = max(0, floor(level0_x0 * scale_x))
        y0 = max(0, floor(level0_y0 * scale_y))
        x1 = min(self.thumbnail.width, ceil(level0_x1 * scale_x))
        y1 = min(self.thumbnail.height, ceil(level0_y1 * scale_y))

        return x0, y0, x1, y1

    def score_tile(self, tile: Tile) -> TileScore:
        x0, y0, x1, y1 = self._thumbnail_bounds(tile)

        if x1 <= x0 or y1 <= y0:
            fraction = 0.0
        else:
            region = self.result.mask[y0:y1, x0:x1]
            fraction = float(np.mean(region > 0))

        return TileScore(
            tile=tile,
            tissue_fraction=fraction,
            is_candidate=fraction >= self.min_tissue_fraction,
        )

    def score_tiles(self, tiles: Iterable[Tile]) -> list[TileScore]:
        return [self.score_tile(tile) for tile in tiles]

    def filter_tiles(self, tiles: Iterable[Tile]) -> Iterator[Tile]:
        for score in self.score_tiles(tiles):
            if score.is_candidate:
                yield score.tile
