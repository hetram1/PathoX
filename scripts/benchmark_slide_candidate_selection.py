from pathlib import Path
import time

import numpy as np
import openslide

from pathox.inference.slide_engine import SlideInferenceEngine
from pathox.native import score_tissue_tiles


IMAGE_ID = "7c9d2571c6ced829b53f37465b030c5b"

SLIDE = Path(
    "/mnt/d/PathoXData/panda/radboud/images/"
    f"{IMAGE_ID}.tiff"
)

TILE_SIZE = 512
STRIDE = 512
THUMBNAIL_SIZE = 2048
TISSUE_THRESHOLD = 0.12
REPEATS = 10


def python_candidates(slide, tissue_mask):
    thumb_h, thumb_w = tissue_mask.shape
    slide_w, slide_h = slide.dimensions

    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h

    candidates = []

    for y in range(
        0,
        slide_h - TILE_SIZE + 1,
        STRIDE,
    ):
        for x in range(
            0,
            slide_w - TILE_SIZE + 1,
            STRIDE,
        ):
            tx0 = int(x * scale_x)
            ty0 = int(y * scale_y)

            tx1 = max(
                tx0 + 1,
                int((x + TILE_SIZE) * scale_x),
            )
            ty1 = max(
                ty0 + 1,
                int((y + TILE_SIZE) * scale_y),
            )

            region = tissue_mask[
                ty0:min(ty1, thumb_h),
                tx0:min(tx1, thumb_w),
            ]

            fraction = (
                float(region.mean())
                if region.size
                else 0.0
            )

            if fraction >= TISSUE_THRESHOLD:
                candidates.append(
                    (
                        x,
                        y,
                        fraction,
                    )
                )

    return candidates


def native_candidates(slide, tissue_mask):
    slide_w, slide_h = slide.dimensions

    native_mask = np.ascontiguousarray(
        tissue_mask,
        dtype=np.uint8,
    )

    result = score_tissue_tiles(
        native_mask,
        slide_w,
        slide_h,
        TILE_SIZE,
        STRIDE,
        TISSUE_THRESHOLD,
    )

    return [
        (
            int(row[0]),
            int(row[1]),
            float(row[2]),
        )
        for row in result
    ]


def main():
    if score_tissue_tiles is None:
        raise RuntimeError(
            "Native score_tissue_tiles is unavailable"
        )

    slide = openslide.OpenSlide(str(SLIDE))

    engine = SlideInferenceEngine.__new__(
        SlideInferenceEngine
    )

    thumbnail = slide.get_thumbnail(
        (THUMBNAIL_SIZE, THUMBNAIL_SIZE)
    )

    tissue_mask = engine._tissue_mask(thumbnail)

    print("=" * 60)
    print("PATHOX WSI CANDIDATE SELECTION BENCHMARK")
    print("=" * 60)

    print("Slide dimensions :", slide.dimensions)
    print("Thumbnail       :", tissue_mask.shape)
    print("Tile size       :", TILE_SIZE)
    print("Stride          :", STRIDE)
    print("Tissue threshold:", TISSUE_THRESHOLD)
    print("Repeats         :", REPEATS)

    # Correctness check first.
    python_result = python_candidates(
        slide,
        tissue_mask,
    )

    native_result = native_candidates(
        slide,
        tissue_mask,
    )

    python_coords = {
        (x, y)
        for x, y, _ in python_result
    }

    native_coords = {
        (x, y)
        for x, y, _ in native_result
    }

    print()
    print("Correctness")
    print("-" * 60)
    print("Python candidates :", len(python_result))
    print("Native candidates :", len(native_result))
    print("Coordinate match  :", python_coords == native_coords)

    if python_coords != native_coords:
        missing = python_coords - native_coords
        extra = native_coords - python_coords

        print("Missing in native :", len(missing))
        print("Extra in native   :", len(extra))

        raise RuntimeError(
            "Native candidate selection does not match Python reference"
        )

    python_fractions = {
        (x, y): fraction
        for x, y, fraction in python_result
    }

    native_fractions = {
        (x, y): fraction
        for x, y, fraction in native_result
    }

    max_fraction_error = max(
        (
            abs(
                python_fractions[key]
                - native_fractions[key]
            )
            for key in python_coords
        ),
        default=0.0,
    )

    print(
        "Max fraction error:",
        f"{max_fraction_error:.8f}",
    )

    if max_fraction_error > 1e-6:
        raise RuntimeError(
            "Native tissue fractions differ from Python reference"
        )

    # Warm up.
    for _ in range(2):
        python_candidates(
            slide,
            tissue_mask,
        )
        native_candidates(
            slide,
            tissue_mask,
        )

    python_times = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        python_candidates(
            slide,
            tissue_mask,
        )

        python_times.append(
            time.perf_counter() - start
        )

    native_times = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        native_candidates(
            slide,
            tissue_mask,
        )

        native_times.append(
            time.perf_counter() - start
        )

    python_mean = float(
        np.mean(python_times)
    )

    native_mean = float(
        np.mean(native_times)
    )

    speedup = (
        python_mean / native_mean
        if native_mean > 0
        else float("inf")
    )

    print()
    print("Performance")
    print("-" * 60)
    print(
        f"Python mean : {python_mean:.6f} s"
    )
    print(
        f"Native mean : {native_mean:.6f} s"
    )
    print(
        f"Speedup     : {speedup:.2f}x"
    )

    print()
    print("PATHOX C++/OPENMP BENCHMARK: PASSED")

    slide.close()


if __name__ == "__main__":
    main()
