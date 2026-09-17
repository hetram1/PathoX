from pathlib import Path
from collections import Counter
from pathox.datasets.panda_segmentation import (
    PANDAImageMaskPair,
    PANDATileExtractor,
)

ID = "00928370e2dfeb8a507667ef1d4efcbb"

root = Path("/mnt/d/PathoXData/panda/radboud")
out = Path("/mnt/d/PathoXData/processed/panda_tiles") / ID

pair = PANDAImageMaskPair(
    image_id=ID,
    image_path=root / "images" / f"{ID}.tiff",
    mask_path=root / "masks" / f"{ID}_mask.tiff",
)

rows, manifest = PANDATileExtractor(
    tile_size=512,
    stride=512,
).extract(pair, out, per_class=8)

print("Extracted tiles:", len(rows))
print("Manifest:", manifest)

print("\nTarget-class counts:")
print(Counter(r["target_class"] for r in rows))

print("\nFirst 10:")
for r in rows[:10]:
    print(
        r["target_class"],
        r["x"],
        r["y"],
        r["classes_present"],
    )
