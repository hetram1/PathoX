from pathlib import Path
import json
import time

import numpy as np
from PIL import Image

from pathox.native import score_tiles, max_threads


MASK = Path(
    "/mnt/d/PathoXData/panda/radboud/"
    "masks/00928370e2dfeb8a507667ef1d4efcbb_mask.tiff"
)

OUT = Path("outputs/panda_acceleration")
OUT.mkdir(parents=True, exist_ok=True)

Image.MAX_IMAGE_PIXELS = None


def python_score_tiles(
    mask,
    tile_size=512,
    stride=512,
    foreground_threshold=0.70,
    class_threshold=0.01,
):
    h, w = mask.shape
    rows = []

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = mask[y:y + tile_size, x:x + tile_size]

            counts = np.bincount(
                tile.reshape(-1),
                minlength=6,
            )

            total = tile_size * tile_size
            background_fraction = counts[0] / total

            if background_fraction >= foreground_threshold:
                rows.append(
                    [
                        x,
                        y,
                        0,
                        background_fraction,
                    ]
                )
                continue

            best_class = int(np.argmax(counts[1:6])) + 1
            fraction = counts[best_class] / total

            if fraction >= class_threshold:
                rows.append(
                    [
                        x,
                        y,
                        best_class,
                        fraction,
                    ]
                )

    return np.asarray(rows, dtype=np.float32)


def main():
    if score_tiles is None:
        raise RuntimeError(
            "Native module is unavailable. Rebuild it first."
        )

    mask = np.asarray(
        Image.open(MASK)
        .convert("RGB")
        .getchannel("R"),
        dtype=np.uint8,
    ).copy()

    print("=" * 60)
    print("PATHOX PYTHON vs C++/OPENMP BENCHMARK")
    print("=" * 60)
    print("Mask shape       :", mask.shape)
    print("OpenMP threads   :", max_threads())

    # Python correctness + timing.
    python_start = time.perf_counter()
    python_result = python_score_tiles(mask)
    python_time = time.perf_counter() - python_start

    # Native correctness + timing.
    native_start = time.perf_counter()
    native_result = score_tiles(
        mask,
        tile_size=512,
        stride=512,
        foreground_threshold=0.70,
        class_threshold=0.01,
    )
    native_time = time.perf_counter() - native_start

    print("\nResults")
    print("Python candidates :", python_result.shape[0])
    print("C++ candidates    :", native_result.shape[0])
    print(f"Python time       : {python_time:.4f}s")
    print(f"C++/OpenMP time   : {native_time:.4f}s")

    if native_time > 0:
        speedup = python_time / native_time
    else:
        speedup = float("inf")

    print(f"Speedup           : {speedup:.2f}x")

    # Verify exact candidate coordinates/classes/fractions.
    assert python_result.shape == native_result.shape
    assert np.array_equal(
        python_result[:, :3],
        native_result[:, :3],
    )
    assert np.allclose(
        python_result[:, 3],
        native_result[:, 3],
        rtol=1e-6,
        atol=1e-6,
    )

    benchmark = {
        "mask_shape": list(mask.shape),
        "openmp_threads": int(max_threads()),
        "python_candidates": int(python_result.shape[0]),
        "cpp_candidates": int(native_result.shape[0]),
        "python_seconds": python_time,
        "cpp_openmp_seconds": native_time,
        "speedup": speedup,
    }

    with (OUT / "benchmark.json").open("w") as f:
        json.dump(benchmark, f, indent=2)

    print("\nCorrectness")
    print("Candidate outputs: MATCH")

    print("\nSaved:")
    print(OUT / "benchmark.json")

    print("\nCPP / OPENMP BENCHMARK: PASSED")


if __name__ == "__main__":
    main()
