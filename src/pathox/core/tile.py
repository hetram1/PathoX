from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from PIL import Image

from .wsi import WSIReader


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    level: int
    width: int
    height: int


class TileExtractor:
    """Generate and read tiles from a whole-slide image."""

    def __init__(
        self,
        reader: WSIReader,
        tile_size: int = 512,
        overlap: int = 0,
    ) -> None:
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")

        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        if overlap >= tile_size:
            raise ValueError("overlap must be smaller than tile_size")

        self.reader = reader
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def _positions(self, length: int, include_partial: bool) -> list[int]:
        if include_partial:
            return list(range(0, length, self.stride))

        if length < self.tile_size:
            return []

        return list(range(0, length - self.tile_size + 1, self.stride))

    def iter_tiles(
        self,
        level: int = 0,
        include_partial: bool = False,
    ) -> Iterator[Tile]:
        if level < 0 or level >= self.reader.slide.level_count:
            raise ValueError(f"Invalid pyramid level: {level}")

        level_width, level_height = self.reader.slide.level_dimensions[level]

        xs = self._positions(level_width, include_partial)
        ys = self._positions(level_height, include_partial)

        for y in ys:
            for x in xs:
                width = min(self.tile_size, level_width - x)
                height = min(self.tile_size, level_height - y)

                yield Tile(
                    x=x,
                    y=y,
                    level=level,
                    width=width,
                    height=height,
                )

    def read_tile(self, tile: Tile) -> Image.Image:
        downsample = self.reader.slide.level_downsamples[tile.level]

        level0_x = round(tile.x * downsample)
        level0_y = round(tile.y * downsample)

        return self.reader.read_region(
            x=level0_x,
            y=level0_y,
            level=tile.level,
            width=tile.width,
            height=tile.height,
        )
