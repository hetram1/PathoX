from pathlib import Path

import pytest

from pathox import TileExtractor, TissueTileFilter, WSIReader


WSI_PATH = Path("data/test/CMU-1-Small-Region.svs")


def test_invalid_fraction() -> None:
    with WSIReader(WSI_PATH) as reader:
        with pytest.raises(ValueError):
            TissueTileFilter(
                reader,
                min_tissue_fraction=1.5,
            )


def test_invalid_thumbnail_width() -> None:
    with WSIReader(WSI_PATH) as reader:
        with pytest.raises(ValueError):
            TissueTileFilter(
                reader,
                thumbnail_width=0,
            )


def test_scores_are_bounded() -> None:
    with WSIReader(WSI_PATH) as reader:
        extractor = TileExtractor(
            reader,
            tile_size=512,
            overlap=128,
        )

        tiles = list(
            extractor.iter_tiles(
                level=0,
                include_partial=True,
            )
        )

        tile_filter = TissueTileFilter(
            reader,
            thumbnail_width=256,
            min_tissue_fraction=0.10,
        )

        scores = tile_filter.score_tiles(tiles)

        assert len(scores) == len(tiles)
        assert all(
            0.0 <= score.tissue_fraction <= 1.0
            for score in scores
        )


def test_filter_returns_only_candidates() -> None:
    with WSIReader(WSI_PATH) as reader:
        extractor = TileExtractor(
            reader,
            tile_size=512,
            overlap=128,
        )

        tiles = list(
            extractor.iter_tiles(
                level=0,
                include_partial=True,
            )
        )

        tile_filter = TissueTileFilter(
            reader,
            thumbnail_width=256,
            min_tissue_fraction=0.10,
        )

        scores = tile_filter.score_tiles(tiles)
        candidates = list(tile_filter.filter_tiles(tiles))

        expected = [
            score.tile
            for score in scores
            if score.is_candidate
        ]

        assert candidates == expected
