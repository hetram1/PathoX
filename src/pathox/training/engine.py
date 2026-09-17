from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from pathox.models import CombinedSegmentationLoss


@dataclass(frozen=True)
class TrainResult:
    initial_loss: float
    final_loss: float
    epochs: int
    samples: int


class SegmentationTrainer:
    """Training engine for semantic segmentation models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        self.model = model.to(device)
        self.device = device

        self.loss_fn = CombinedSegmentationLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

    def train(
        self,
        loader: DataLoader,
        epochs: int = 1,
    ) -> TrainResult:
        if epochs <= 0:
            raise ValueError(
                "epochs must be positive"
            )

        self.model.train()

        losses: list[float] = []

        for _ in range(epochs):
            for images, masks in loader:
                images = images.to(
                    self.device,
                    non_blocking=True,
                )

                masks = masks.to(
                    self.device,
                    non_blocking=True,
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                logits = self.model(images)

                loss = self.loss_fn(
                    logits,
                    masks,
                )

                loss.backward()

                self.optimizer.step()

                losses.append(
                    float(loss.detach().cpu())
                )

        if not losses:
            raise RuntimeError(
                "No training batches were processed"
            )

        return TrainResult(
            initial_loss=losses[0],
            final_loss=losses[-1],
            epochs=epochs,
            samples=len(loader.dataset),
        )

    def save_checkpoint(
        self,
        path: str,
    ) -> None:
        torch.save(
            {
                "model_state_dict": (
                    self.model.state_dict()
                ),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),
            },
            path,
        )
