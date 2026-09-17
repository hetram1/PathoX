from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image

from pathox.datasets.segmentation_dataset import PANDASegmentationDataset
from pathox.models.unet import UNet
from pathox.models.losses import CombinedSegmentationLoss


MANIFEST = Path(
    "/mnt/d/PathoXData/processed/panda_tiles/"
    "panda_segmentation_manifest.csv"
)

CHECKPOINT_DIR = Path("/mnt/d/PathoXData/experiments/panda")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
EPOCHS = 12
LR = 2e-4
NUM_CLASSES = 6


def compute_class_weights(dataset_manifest, num_classes=6):
    manifest = pd.read_csv(dataset_manifest)
    manifest = manifest[manifest["split"] == "train"]

    counts = np.zeros(num_classes, dtype=np.int64)

    for mask_path in manifest["mask_path"]:
        mask = np.asarray(
            Image.open(mask_path).convert("RGB")
        )[:, :, 0]

        counts += np.bincount(
            mask.ravel(),
            minlength=num_classes,
        )

    frequencies = counts / counts.sum()

    median_frequency = np.median(
        frequencies[frequencies > 0]
    )

    weights = np.sqrt(
        median_frequency / frequencies
    )

    weights /= weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def segmentation_metrics(pred, target, num_classes=6):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    dice = []
    iou = []
    f1 = []

    for cls in range(num_classes):
        p = pred == cls
        t = target == cls

        tp = np.logical_and(p, t).sum()
        fp = np.logical_and(p, ~t).sum()
        fn = np.logical_and(~p, t).sum()

        denom_dice = 2 * tp + fp + fn
        denom_iou = tp + fp + fn

        d = (2 * tp / denom_dice) if denom_dice else np.nan
        j = (tp / denom_iou) if denom_iou else np.nan

        dice.append(d)
        iou.append(j)
        f1.append(d)

    return np.array(dice), np.array(iou), np.array(f1)


def main():
    print("Device:", DEVICE)

    train_ds = PANDASegmentationDataset(
        MANIFEST,
        split="train",
        augment=True,
    )

    val_ds = PANDASegmentationDataset(
        MANIFEST,
        split="val",
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("Train tiles:", len(train_ds))
    print("Val tiles  :", len(val_ds))

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=16,
    ).to(DEVICE)

    class_weights = compute_class_weights(
        MANIFEST,
        NUM_CLASSES,
    ).to(DEVICE)

    print(
        "Class weights:",
        class_weights.detach().cpu().numpy(),
    )

    criterion = CombinedSegmentationLoss(
        dice_weight=0.5,
        class_weights=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    history = []
    best_mean_dice = -float("inf")

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(
                DEVICE,
                non_blocking=True,
            )
            masks = masks.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0

        all_pred = []
        all_target = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(
                    DEVICE,
                    non_blocking=True,
                )
                masks = masks.to(
                    DEVICE,
                    non_blocking=True,
                )

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss += loss.item() * images.size(0)

                pred = torch.argmax(logits, dim=1)

                all_pred.append(pred)
                all_target.append(masks)

        val_loss /= len(val_ds)

        predictions = torch.cat(all_pred, dim=0)
        targets = torch.cat(all_target, dim=0)

        dice, iou, f1 = segmentation_metrics(
            predictions,
            targets,
            NUM_CLASSES,
        )

        valid_dice = dice[~np.isnan(dice)]
        valid_iou = iou[~np.isnan(iou)]

        mean_dice = float(valid_dice.mean())
        mean_iou = float(valid_iou.mean())

        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "class_dice": dice.tolist(),
            "class_iou": iou.tolist(),
            "seconds": elapsed,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"dice={mean_dice:.4f} | "
            f"IoU={mean_iou:.4f} | "
            f"{elapsed:.1f}s"
        )

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "mean_dice": mean_dice,
                "mean_iou": mean_iou,
                "num_classes": NUM_CLASSES,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "pathox_panda_weighted_best.pt",
            )

            print("  saved best checkpoint")

    with (CHECKPOINT_DIR / "training_history_weighted.json").open("w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete")
    print("Best validation Dice:", best_mean_dice)
    print(
        "Checkpoint:",
        CHECKPOINT_DIR / "pathox_panda_weighted_best.pt",
    )


if __name__ == "__main__":
    main()
