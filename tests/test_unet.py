import pytest
import torch

from pathox import UNet


def test_invalid_class_count() -> None:
    with pytest.raises(ValueError):
        UNet(num_classes=1)


def test_forward_shape_cpu() -> None:
    model = UNet(
        in_channels=3,
        num_classes=6,
        base_channels=16,
    )

    x = torch.randn(
        1,
        3,
        256,
        256,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 6, 256, 256)


def test_model_has_trainable_parameters() -> None:
    model = UNet(
        num_classes=6,
        base_channels=16,
    )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(parameters) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_forward_gpu() -> None:
    model = UNet(
        num_classes=6,
        base_channels=16,
    ).cuda()

    x = torch.randn(
        1,
        3,
        256,
        256,
        device="cuda",
    )

    with torch.no_grad():
        output = model(x)

    assert output.device.type == "cuda"
    assert output.shape == (1, 6, 256, 256)
