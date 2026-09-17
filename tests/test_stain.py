import numpy as np
from PIL import Image

from pathox import MacenkoNormalizer, StainAugmenter


def test_macenko_preserves_shape() -> None:
    pixels = np.zeros((128, 128, 3), dtype=np.uint8)
    pixels[:, :, 0] = 180
    pixels[:, :, 1] = 100
    pixels[:, :, 2] = 140

    image = Image.fromarray(pixels)

    result = MacenkoNormalizer().normalize(image)

    assert result.image.size == image.size
    assert result.input_shape == image.size
    assert result.output_shape == image.size


def test_macenko_output_is_valid_rgb() -> None:
    pixels = np.random.default_rng(42).integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )

    image = Image.fromarray(pixels)

    result = MacenkoNormalizer().normalize(image)

    output = np.asarray(result.image)

    assert output.dtype == np.uint8
    assert output.shape == (64, 64, 3)
    assert output.min() >= 0
    assert output.max() <= 255


def test_augmenter_preserves_shape() -> None:
    pixels = np.full(
        (64, 64, 3),
        150,
        dtype=np.uint8,
    )

    image = Image.fromarray(pixels)

    augmented = StainAugmenter().apply(
        image,
        rng=np.random.default_rng(42),
    )

    assert augmented.size == image.size


def test_negative_augmentation_is_rejected() -> None:
    try:
        StainAugmenter(brightness=-0.1)
    except ValueError:
        return

    raise AssertionError("Expected ValueError")
