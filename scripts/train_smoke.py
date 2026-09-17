from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pathox.data import (
    SyntheticSegmentationConfig,
    SyntheticSegmentationDataset,
)
from pathox.models import UNet
from pathox.training import SegmentationTrainer


CHECKPOINT = Path(
    "outputs/test/pathox_smoke_checkpoint.pt"
)


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = SyntheticSegmentationDataset(
        SyntheticSegmentationConfig(
            samples=24,
            image_size=128,
            num_classes=6,
            seed=42,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = UNet(
        in_channels=3,
        num_classes=6,
        base_channels=16,
    )

    trainer = SegmentationTrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
    )

    result = trainer.train(
        loader,
        epochs=3,
    )

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_checkpoint(
        str(CHECKPOINT)
    )

    print("=== PathoX Training Smoke Test ===")
    print(f"Device: {device}")
    print(f"Samples: {result.samples}")
    print(f"Epochs: {result.epochs}")
    print(
        f"Initial loss: "
        f"{result.initial_loss:.6f}"
    )
    print(
        f"Final loss: "
        f"{result.final_loss:.6f}"
    )
    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    if result.final_loss >= result.initial_loss:
        raise RuntimeError(
            "Training loss did not decrease"
        )

    print("Training pipeline OK.")


if __name__ == "__main__":
    main()
