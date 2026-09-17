from pathlib import Path
import csv
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from pathox.datasets.segmentation_dataset import PANDASegmentationDataset
from pathox.models.unet import UNet


MANIFEST = Path(
    "/mnt/d/PathoXData/processed/panda_tiles/"
    "panda_segmentation_manifest.csv"
)

CHECKPOINT = Path(
    "/mnt/d/PathoXData/experiments/panda/"
    "pathox_panda_best.pt"
)

OUT = Path("outputs/panda_evaluation")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6


def metrics_from_confusion(cm):
    results = {}

    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        dice_den = 2 * tp + fp + fn
        iou_den = tp + fp + fn

        dice = float(2 * tp / dice_den) if dice_den else None
        iou = float(tp / iou_den) if iou_den else None

        results[c] = {
            "dice": dice,
            "iou": iou,
            "f1": dice,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    valid_dice = [
        x["dice"] for x in results.values()
        if x["dice"] is not None
    ]
    valid_iou = [
        x["iou"] for x in results.values()
        if x["iou"] is not None
    ]

    return results, float(np.mean(valid_dice)), float(np.mean(valid_iou))


def make_overlay(image, pred, target):
    image = np.asarray(image.convert("RGB")).copy()

    # Prediction shown as translucent red.
    pred_fg = pred > 0
    image[pred_fg] = (
        0.65 * image[pred_fg] +
        0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # Ground truth shown by bright green boundary.
    boundary = np.zeros_like(target, dtype=bool)

    boundary[1:, :] |= target[1:, :] != target[:-1, :]
    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]

    image[boundary] = np.array([0, 255, 0], dtype=np.uint8)

    return Image.fromarray(image)


def save_prediction_examples(dataset, model, count=8):
    example_dir = OUT / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    saved = 0

    with torch.no_grad():
        for image_tensor, mask_tensor in loader:
            image_tensor = image_tensor.to(DEVICE)

            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
            target = mask_tensor[0].numpy()

            image_np = (
                image_tensor[0]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
                * 255
            ).clip(0, 255).astype(np.uint8)

            image = Image.fromarray(image_np)
            truth = Image.fromarray(target.astype(np.uint8))
            prediction = Image.fromarray(pred.astype(np.uint8))
            overlay = make_overlay(image, pred, target)

            image.save(example_dir / f"example_{saved:02d}_image.png")
            truth.save(example_dir / f"example_{saved:02d}_truth.png")
            prediction.save(
                example_dir / f"example_{saved:02d}_prediction.png"
            )
            overlay.save(
                example_dir / f"example_{saved:02d}_overlay.png"
            )

            saved += 1

            if saved >= count:
                break


def make_slide_heatmap(records, predictions):
    by_slide = {}

    for r, pred in zip(records, predictions):
        image_id = r["image_id"]
        x = int(r["x"])
        y = int(r["y"])

        score = float((pred > 0).mean())

        by_slide.setdefault(image_id, []).append(
            (x, y, score)
        )

    heatmap_dir = OUT / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for image_id, points in by_slide.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        cell = 512

        width = (max_x - min_x) // cell + 1
        height = (max_y - min_y) // cell + 1

        canvas = Image.new(
            "RGB",
            (width * cell, height * cell),
            "white",
        )

        draw = ImageDraw.Draw(canvas)

        for x, y, score in points:
            gx = (x - min_x) // cell
            gy = (y - min_y) // cell

            # White -> red intensity.
            value = int(255 * (1.0 - score))
            fill = (255, value, value)

            draw.rectangle(
                [
                    gx * cell,
                    gy * cell,
                    (gx + 1) * cell - 1,
                    (gy + 1) * cell - 1,
                ],
                fill=fill,
            )

        canvas.save(
            heatmap_dir / f"{image_id}_heatmap.png"
        )


def main():
    print("Device:", DEVICE)

    dataset = PANDASegmentationDataset(
        MANIFEST,
        split="val",
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
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

    cm = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    records = dataset.records
    predictions_for_heatmap = []

    offset = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)

            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            masks = masks.numpy()

            for pred, target in zip(preds, masks):
                for t, p in zip(
                    target.reshape(-1),
                    pred.reshape(-1),
                ):
                    cm[t, p] += 1

                predictions_for_heatmap.append(pred)

            offset += len(preds)

    class_metrics, mean_dice, mean_iou = metrics_from_confusion(cm)

    metrics = {
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT),
        "validation_tiles": len(dataset),
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
    }

    with (OUT / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    with (OUT / "per_class_metrics.csv").open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "class",
                "dice",
                "iou",
                "f1",
                "tp",
                "fp",
                "fn",
            ]
        )

        for c in range(NUM_CLASSES):
            m = class_metrics[c]
            writer.writerow(
                [
                    c,
                    m["dice"],
                    m["iou"],
                    m["f1"],
                    m["tp"],
                    m["fp"],
                    m["fn"],
                ]
            )

    print("\nValidation results")
    print("=" * 50)
    print("Validation tiles :", len(dataset))
    print(f"Mean Dice        : {mean_dice:.4f}")
    print(f"Mean IoU         : {mean_iou:.4f}")

    print("\nPer-class metrics")
    for c in range(NUM_CLASSES):
        m = class_metrics[c]
        print(
            f"class {c}: "
            f"Dice={m['dice']} "
            f"IoU={m['iou']}"
        )

    save_prediction_examples(
        dataset,
        model,
        count=8,
    )

    make_slide_heatmap(
        records,
        predictions_for_heatmap,
    )

    print("\nSaved:")
    print(OUT / "metrics.json")
    print(OUT / "per_class_metrics.csv")
    print(OUT / "examples")
    print(OUT / "heatmaps")

    print("\nPANDA MODEL EVALUATION: PASSED")


if __name__ == "__main__":
    main()
