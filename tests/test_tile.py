from pathlib import Path

import pytest

from pathox import TileExtractor, WSIReader


WSI_PATH = Path("data/test/CMU-1-Small-Region.svs")


def test_invalid_tile_size() -> None:
    with WSIReader(WSI_PATH) as reader:
        with pytest.raises(ValueError):
            TileExtractor(reader, tile_size=0)


def test_invalid_overlap() -> None:
    with WSIReader(WSI_PATH) as reader:
        with pytest.raises(ValueError):
            TileExtractor(reader, tile_size=512, overlap=512)


def test_full_tiles() -> None:
    with WSIReader(WSI_PATH) as reader:
        extractor = TileExtractor(reader, tile_size=512)
        tiles = list(extractor.iter_tiles(level=0))

        assert len(tiles) == 20
        assert all(tile.width == 512 for tile in tiles)
        assert all(tile.height == 512 for tile in tiles)


def test_partial_tiles() -> None:
    with WSIReader(WSI_PATH) as reader:
        extractor = TileExtractor(reader, tile_size=512)
        tiles = list(extractor.iter_tiles(level=0, include_partial=True))

        assert len(tiles) == 30
        assert tiles[0].width == 512
        assert tiles[0].height == 512
        assert tiles[-1].width < 512
        assert tiles[-1].height < 512


def test_read_tile() -> None:
    with WSIReader(WSI_PATH) as reader:
        extractor = TileExtractor(reader, tile_size=512)
        tile = next(extractor.iter_tiles())
        image = extractor.read_tile(tile)

        assert image.size == (512, 512)
