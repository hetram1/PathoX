from pathlib import Path
import torch
from torch.utils.data import DataLoader

from pathox.datasets.segmentation_dataset import PANDASegmentationDataset

manifest = Path(
    "/mnt/d/PathoXData/processed/panda_tiles/"
    "00928370e2dfeb8a507667ef1d4efcbb/manifest.csv"
)

dataset = PANDASegmentationDataset(manifest, augment=True)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

images, masks = next(iter(loader))

print("Dataset size :", len(dataset))
print("Image batch  :", tuple(images.shape), images.dtype)
print("Mask batch   :", tuple(masks.shape), masks.dtype)
print("Image range  :", float(images.min()), float(images.max()))
print("Mask labels  :", torch.unique(masks).tolist())

assert images.shape == (4, 3, 512, 512)
assert masks.shape == (4, 512, 512)
assert images.dtype == torch.float32
assert masks.dtype == torch.int64
assert set(torch.unique(masks).tolist()).issubset({0, 1, 2, 3, 4, 5})

print("\nREAL PANDA DATASET CHECK: PASSED")
