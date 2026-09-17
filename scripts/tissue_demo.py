from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pathox import TissueDetector, WSIReader


WSI_PATH = Path("data/test/CMU-1-Small-Region.svs")
OUTPUT_DIR = Path("outputs/test/tissue")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with WSIReader(WSI_PATH) as reader:
        thumbnail = reader.thumbnail(800)

    detector = TissueDetector()
    result = detector.detect(thumbnail)

    mask_path = OUTPUT_DIR / "tissue_mask.png"
    Image.fromarray(result.mask).save(mask_path)

    rgb = np.asarray(thumbnail.convert("RGB"))
    mask = result.mask > 0

    overlay = rgb.copy()
    overlay[mask] = (
        0.5 * overlay[mask]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    overlay_path = OUTPUT_DIR / "tissue_overlay.png"
    Image.fromarray(overlay).save(overlay_path)

    print("=== PathoX Tissue Detection ===")
    print(f"Thumbnail size: {thumbnail.size}")
    print(f"Tissue fraction: {result.tissue_fraction:.4f}")
    print(f"Mask: {mask_path}")
    print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
