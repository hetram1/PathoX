from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from pathox.native import score_tiles, max_threads


MASK = Path(
    "/mnt/d/PathoXData/panda/radboud/"
    "masks/00928370e2dfeb8a507667ef1d4efcbb_mask.tiff"
)

if score_tiles is None:
    raise RuntimeError("Native PathoX module could not be imported")

Image.MAX_IMAGE_PIXELS = None

mask = np.asarray(
    Image.open(MASK).convert("RGB").getchannel("R"),
    dtype=np.uint8,
).copy()

print("Mask shape :", mask.shape)
print("OpenMP threads:", max_threads())

start = time.perf_counter()

result = score_tiles(
    mask,
    tile_size=512,
    stride=512,
    foreground_threshold=0.70,
    class_threshold=0.01,
)

elapsed = time.perf_counter() - start

print("Candidates :", result.shape[0])
print("Columns    :", result.shape[1])
print("Time       :", f"{elapsed:.3f}s")

print("\nFirst 10 candidates:")
for row in result[:10]:
    print(
        f"x={int(row[0])} "
        f"y={int(row[1])} "
        f"class={int(row[2])} "
        f"fraction={row[3]:.4f}"
    )

assert result.ndim == 2
assert result.shape[1] == 4
assert np.all(result[:, 2] >= 0)
assert np.all(result[:, 2] <= 5)
assert np.all(result[:, 3] >= 0)
assert np.all(result[:, 3] <= 1)

print("\nC++ / OPENMP ACCELERATION CHECK: PASSED")
