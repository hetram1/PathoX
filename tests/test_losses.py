import pytest
import torch

from pathox.models import CombinedSegmentationLoss, DiceLoss


def test_dice_loss_is_zero_for_perfect_prediction() -> None:
    target = torch.tensor(
        [[[0, 1], [2, 1]]],
        dtype=torch.long,
    )

    logits = torch.full(
        (1, 3, 2, 2),
        -20.0,
    )

    for y in range(2):
        for x in range(2):
            logits[
                0,
                target[0, y, x],
                y,
                x,
            ] = 20.0

    loss = DiceLoss()(logits, target)

    assert loss.item() < 1e-5


def test_combined_loss_is_finite() -> None:
    logits = torch.randn(
        2,
        6,
        32,
        32,
    )

    target = torch.randint(
        0,
        6,
        (2, 32, 32),
    )

    loss = CombinedSegmentationLoss()(
        logits,
        target,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_loss_supports_backpropagation() -> None:
    logits = torch.randn(
        1,
        6,
        32,
        32,
        requires_grad=True,
    )

    target = torch.randint(
        0,
        6,
        (1, 32, 32),
    )

    loss = CombinedSegmentationLoss()(
        logits,
        target,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_invalid_dice_weight() -> None:
    with pytest.raises(ValueError):
        CombinedSegmentationLoss(
            dice_weight=1.5,
        )
