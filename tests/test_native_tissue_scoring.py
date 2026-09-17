import numpy as np
import pytest

from pathox.native import score_tissue_tiles


@pytest.mark.skipif(
    score_tissue_tiles is None,
    reason="Native PathoX extension is not built",
)
def test_score_tissue_tiles_matches_expected_tiles():
    mask = np.zeros((8, 8), dtype=np.uint8)

    mask[:4, :4] = 1
    mask[4:, 4:] = 1

    result = score_tissue_tiles(
        mask,
        slide_width=8,
        slide_height=8,
        tile_size=4,
        stride=4,
        min_tissue_fraction=0.5,
    )

    assert result.shape == (2, 3)

    coordinates = {
        (int(row[0]), int(row[1]))
        for row in result
    }

    assert coordinates == {
        (0, 0),
        (4, 4),
    }

    fractions = {
        (int(row[0]), int(row[1])): float(row[2])
        for row in result
    }

    assert fractions[(0, 0)] == pytest.approx(1.0)
    assert fractions[(4, 4)] == pytest.approx(1.0)


@pytest.mark.skipif(
    score_tissue_tiles is None,
    reason="Native PathoX extension is not built",
)
def test_score_tissue_tiles_rejects_invalid_fraction():
    mask = np.ones((8, 8), dtype=np.uint8)

    with pytest.raises(RuntimeError):
        score_tissue_tiles(
            mask,
            slide_width=8,
            slide_height=8,
            tile_size=4,
            stride=4,
            min_tissue_fraction=1.5,
        )
