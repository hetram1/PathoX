import numpy as np
from PIL import Image

from pathox import TissueDetector


def test_white_background_is_not_tissue() -> None:
    image = Image.fromarray(
        np.full((100, 100, 3), 255, dtype=np.uint8)
    )

    result = TissueDetector().detect(image)

    assert result.tissue_fraction == 0.0
    assert not result.mask.any()


def test_colored_region_is_detected() -> None:
    pixels = np.full((100, 100, 3), 255, dtype=np.uint8)
    pixels[25:75, 25:75] = [150, 70, 120]

    image = Image.fromarray(pixels)

    result = TissueDetector(
        min_component_area=20,
    ).detect(image)

    assert result.tissue_fraction > 0.20
    assert result.mask[50, 50] == 255


def test_kernel_must_be_odd() -> None:
    try:
        TissueDetector(kernel_size=4)
    except ValueError:
        return

    raise AssertionError("Expected ValueError")
