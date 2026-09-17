import time

import torch

from pathox import UNet


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")

    model = UNet(
        in_channels=3,
        num_classes=6,
        base_channels=16,
    ).to(device)

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    x = torch.rand(
        1,
        3,
        512,
        512,
        device=device,
    )

    with torch.inference_mode():
        for _ in range(3):
            _ = model(x)

        torch.cuda.synchronize()

        start = time.perf_counter()

        output = model(x)

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    prediction = output.argmax(dim=1)

    print("=== PathoX GPU Segmentation Test ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Input: {tuple(x.shape)}")
    print(f"Output: {tuple(output.shape)}")
    print(f"Prediction: {tuple(prediction.shape)}")
    print(f"Inference time: {elapsed * 1000:.2f} ms")
    print("GPU segmentation inference OK.")


if __name__ == "__main__":
    main()
