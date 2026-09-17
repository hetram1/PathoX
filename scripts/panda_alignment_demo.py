from pathlib import Path
import numpy as np
from PIL import Image
import openslide

Image.MAX_IMAGE_PIXELS = None

ID = "00928370e2dfeb8a507667ef1d4efcbb"
root = Path("/mnt/d/PathoXData/panda/radboud")
slide_path = root / "images" / f"{ID}.tiff"
mask_path = root / "masks" / f"{ID}_mask.tiff"
out = Path("outputs/test/panda_alignment")
out.mkdir(parents=True, exist_ok=True)

wsi = openslide.OpenSlide(str(slide_path))
mask = Image.open(mask_path).convert("RGB")
a = np.asarray(mask)

# PANDA mask class is encoded in the red channel.
labels = a[:, :, 0]
assert np.all(a[:, :, 1] == 0)
assert np.all(a[:, :, 2] == 0)
assert set(np.unique(labels)).issubset({0, 1, 2, 3, 4, 5})

assert wsi.dimensions == mask.size

print("WSI :", wsi.dimensions)
print("MASK:", mask.size)
print("Labels:", np.unique(labels).tolist())

# Find first 1024x1024 region containing tissue labels.
tile = 1024
chosen = None

for y in range(0, mask.height, tile):
    for x in range(0, mask.width, tile):
        h = min(tile, mask.height - y)
        w = min(tile, mask.width - x)
        m = labels[y:y+h, x:x+w]
        if np.any(m > 0):
            chosen = x, y, w, h, m
            break
    if chosen:
        break

x, y, w, h, m = chosen

img = np.asarray(wsi.read_region((x, y), 0, (w, h)).convert("RGB"))

# Save image and mask.
Image.fromarray(img).save(out / "wsi.png")
Image.fromarray(m.astype(np.uint8)).save(out / "mask.png")

# Red overlay on labeled pixels.
overlay = img.copy()
fg = m > 0
overlay[fg] = (0.55 * overlay[fg] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
Image.fromarray(overlay).save(out / "overlay.png")

print(f"Region: x={x}, y={y}, size={w}x{h}")

print("\nClass counts:")
for v, c in zip(*np.unique(m, return_counts=True)):
    print(f"  {int(v)}: {int(c):,}")

print("\nSaved:")
print(out / "wsi.png")
print(out / "mask.png")
print(out / "overlay.png")
print("\nALIGNMENT CHECK: PASSED")

wsi.close()
