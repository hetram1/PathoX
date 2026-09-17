from pathlib import Path
import json
import time

import numpy as np
import torch

from pathox.inference.slide_engine import SlideInferenceEngine


IMAGE_ID = "7c9d2571c6ced829b53f37465b030c5b"

SLIDE = Path(
    "/mnt/d/PathoXData/panda/radboud/images/"
    f"{IMAGE_ID}.tiff"
)

CHECKPOINT = Path(
    "/mnt/d/PathoXData/experiments/panda/"
    "pathox_panda_sampler_best.pt"
)

OUT = Path(
    "outputs/panda_benchmark"
)

REPEATS = 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PATHOX END-TO-END INFERENCE BENCHMARK")
    print("=" * 60)

    print("Device:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available()
          else "CPU")

    engine = SlideInferenceEngine(
        checkpoint_path=CHECKPOINT,
        tile_size=512,
        stride=512,
        thumbnail_size=2048,
        tissue_threshold=0.12,
        batch_size=4,
        uncertainty_top_k=50,
    )

    times = []

    # Warmup.
    print("\nWarmup...")
    engine.infer(
        SLIDE,
        OUT / "warmup",
        image_id=IMAGE_ID,
        save_hard_tiles=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs.
    for i in range(REPEATS):
        run_dir = OUT / f"run_{i+1}"

        start = time.perf_counter()

        result = engine.infer(
            SLIDE,
            run_dir,
            image_id=IMAGE_ID,
            save_hard_tiles=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(
            f"Run {i + 1}: "
            f"{elapsed:.4f} s | "
            f"{result.analyzed_tiles} tiles"
        )

    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    tiles = result.analyzed_tiles

    tiles_per_second = (
        tiles / mean_time
        if mean_time > 0
        else 0.0
    )

    metrics = {
        "slide": str(SLIDE),
        "checkpoint": str(CHECKPOINT),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "repeats": REPEATS,
        "times_seconds": times,
        "mean_seconds": mean_time,
        "std_seconds": std_time,
        "tiles": tiles,
        "tiles_per_second": tiles_per_second,
        "tissue_fraction": result.tissue_fraction,
        "lesion_fraction": result.lesion_fraction,
        "mean_confidence": result.mean_confidence,
        "mean_entropy": result.mean_entropy,
        "mean_disagreement": result.mean_disagreement,
    }

    with (OUT / "benchmark.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("Summary")
    print("-" * 60)
    print(f"Mean time       : {mean_time:.4f} s")
    print(f"Std deviation   : {std_time:.4f} s")
    print(f"Tiles analyzed  : {tiles}")
    print(f"Tiles / second  : {tiles_per_second:.2f}")
    print(f"Tissue fraction : {result.tissue_fraction:.4f}")
    print(f"Lesion fraction : {result.lesion_fraction:.4f}")
    print(f"Confidence      : {result.mean_confidence:.4f}")
    print(f"Entropy         : {result.mean_entropy:.4f}")
    print(f"Disagreement    : {result.mean_disagreement:.4f}")

    print()
    print("Saved:", OUT / "benchmark.json")
    print()
    print("PATHOX END-TO-END BENCHMARK: PASSED")


if __name__ == "__main__":
    main()
