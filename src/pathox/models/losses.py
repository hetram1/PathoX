from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss."""

    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError("smooth must be positive")

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must have shape [N,C,H,W]")

        if target.ndim != 3:
            raise ValueError("target must have shape [N,H,W]")

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        num_classes = logits.shape[1]

        target_one_hot = torch.nn.functional.one_hot(
            target.long(),
            num_classes=num_classes,
        )

        target_one_hot = target_one_hot.permute(
            0,
            3,
            1,
            2,
        ).to(probabilities.dtype)

        intersection = (
            probabilities * target_one_hot
        ).sum(dim=(0, 2, 3))

        denominator = (
            probabilities.sum(dim=(0, 2, 3))
            + target_one_hot.sum(dim=(0, 2, 3))
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """Cross-entropy plus Dice loss."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError(
                "dice_weight must be in [0, 1]"
            )

        if class_weights is not None:
            if class_weights.ndim != 1:
                raise ValueError(
                    "class_weights must be a 1D tensor"
                )
            self.register_buffer(
                "class_weights",
                class_weights.float(),
            )
        else:
            self.class_weights = None

        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
        )

        dice = self.dice(
            logits,
            target,
        )

        return (
            (1.0 - self.dice_weight) * ce
            + self.dice_weight * dice
        )
