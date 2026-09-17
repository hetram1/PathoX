from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from pathox.datasets.segmentation_dataset import PANDASegmentationDataset
from pathox.models.unet import UNet
from pathox.inference.uncertainty import UncertaintyAnalyzer


MANIFEST = Path(
    "/mnt/d/PathoXData/processed/panda_tiles/"
    "panda_segmentation_manifest.csv"
)

CHECKPOINT = Path(
    "/mnt/d/PathoXData/experiments/panda/"
    "pathox_panda_best.pt"
)

OUT = Path("outputs/panda_uncertainty")
OUT.mkdir(parents=True, exist_ok=True)

HARD = OUT / "hard_tiles"
HARD.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 6
TOP_K = 20


def main():
    print("Device:", DEVICE)

    dataset = PANDASegmentationDataset(
        MANIFEST,
        split="val",
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=16,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    analyzer = UncertaintyAnalyzer()
    results = []

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(DEVICE)

        batch_results = analyzer.analyze(
            model,
            images,
        )

        start = batch_idx * loader.batch_size

        for j, result in enumerate(batch_results):
            record = dataset.records[start + j]

            results.append({
                "image_id": record["image_id"],
                "x": int(record["x"]),
                "y": int(record["y"]),
                "mean_entropy": result.mean_entropy,
                "mean_confidence": result.mean_confidence,
                "disagreement": result.disagreement,
                "image_path": record["image_path"],
                "mask_path": record["mask_path"],
            })

    results.sort(
        key=lambda r: (
            r["mean_entropy"],
            r["disagreement"],
        ),
        reverse=True,
    )

    csv_path = OUT / "uncertainty_scores.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    # Save the hardest tiles for inspection.
    for rank, record in enumerate(results[:TOP_K], start=1):
        src = Path(record["image_path"])
        image = Image.open(src).convert("RGB")
        image.save(
            HARD /
            f"rank_{rank:02d}_"
            f"{record['image_id']}_"
            f"x{record['x']}_y{record['y']}.png"
        )

    print("\n" + "=" * 60)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 60)

    print("Validation tiles :", len(results))
    print("Top hard tiles   :", min(TOP_K, len(results)))

    print("\nMost uncertain tiles:")
    for rank, r in enumerate(results[:10], start=1):
        print(
            f"{rank:02d}. "
            f"slide={r['image_id'][:12]} "
            f"x={r['x']} y={r['y']} "
            f"entropy={r['mean_entropy']:.4f} "
            f"confidence={r['mean_confidence']:.4f} "
            f"disagreement={r['disagreement']:.4f}"
        )

    print("\nSaved:")
    print(csv_path)
    print(HARD)

    print("\nUNCERTAINTY PIPELINE: PASSED")


if __name__ == "__main__":
    main()
