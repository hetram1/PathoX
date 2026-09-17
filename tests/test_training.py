import torch
from torch.utils.data import DataLoader

from pathox.data import (
    SyntheticSegmentationConfig,
    SyntheticSegmentationDataset,
)
from pathox.models import UNet
from pathox.training import SegmentationTrainer


def test_synthetic_dataset_shape() -> None:
    dataset = SyntheticSegmentationDataset(
        SyntheticSegmentationConfig(
            samples=4,
            image_size=64,
            num_classes=6,
        )
    )

    image, mask = dataset[0]

    assert image.shape == (3, 64, 64)
    assert mask.shape == (64, 64)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.int64


def test_training_engine_runs() -> None:
    dataset = SyntheticSegmentationDataset(
        SyntheticSegmentationConfig(
            samples=4,
            image_size=64,
            num_classes=6,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    trainer = SegmentationTrainer(
        UNet(
            num_classes=6,
            base_channels=8,
        ),
        device=torch.device("cpu"),
    )

    result = trainer.train(
        loader,
        epochs=1,
    )

    assert result.samples == 4
    assert result.initial_loss > 0
    assert result.final_loss > 0
