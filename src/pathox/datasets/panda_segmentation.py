from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import openslide


@dataclass(frozen=True)
class PANDAImageMaskPair:
    image_id: str
    image_path: Path
    mask_path: Path


class PANDATileExtractor:
    def __init__(self, tile_size=512, stride=512):
        self.tile_size = tile_size
        self.stride = stride

    @staticmethod
    def _mask_array(mask_path: Path):
        Image.MAX_IMAGE_PIXELS = None
        mask = Image.open(mask_path).convert("RGB")
        assert np.all(np.asarray(mask.getchannel("G")) == 0)
        assert np.all(np.asarray(mask.getchannel("B")) == 0)
        return mask.size, np.asarray(mask.getchannel("R"), dtype=np.uint8)

    def extract(
        self,
        pair: PANDAImageMaskPair,
        output_dir: Path,
        per_class=8,
        classes=(0, 1, 2, 3, 4, 5),
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        size, mask = self._mask_array(pair.mask_path)

        wsi = openslide.OpenSlide(str(pair.image_path))
        assert wsi.dimensions == size

        candidates = {c: [] for c in classes}
        h_img, w_img = size[1], size[0]

        for y in range(0, h_img - self.tile_size + 1, self.stride):
            for x in range(0, w_img - self.tile_size + 1, self.stride):
                m = mask[y:y+self.tile_size, x:x+self.tile_size]
                vals, counts = np.unique(m, return_counts=True)
                freq = dict(zip(vals.tolist(), counts.tolist()))
                total = self.tile_size * self.tile_size

                for c in classes:
                    frac = freq.get(c, 0) / total
                    if c == 0:
                        score = 1.0 - sum(
                            freq.get(k, 0) for k in classes if k != 0
                        ) / total
                        if frac < 0.70:
                            continue
                    else:
                        score = frac
                        if frac < 0.01:
                            continue

                    candidates[c].append((score, x, y, frac, vals.tolist()))

        selected = []
        used = set()

        for c in classes:
            ranked = sorted(candidates[c], reverse=True)
            taken = 0
            for score, x, y, frac, vals in ranked:
                key = (x, y)
                if key in used:
                    continue

                selected.append((c, score, x, y, frac, vals))
                used.add(key)
                taken += 1

                if taken >= per_class:
                    break

        rows = []

        for idx, (target_class, score, x, y, frac, vals) in enumerate(selected):
            img = wsi.read_region(
                (x, y), 0, (self.tile_size, self.tile_size)
            ).convert("RGB")

            m = Image.fromarray(
                mask[y:y+self.tile_size, x:x+self.tile_size]
            )

            stem = f"{pair.image_id}_{idx:03d}_x{x}_y{y}"

            image_file = output_dir / f"{stem}.png"
            mask_file = output_dir / f"{stem}_mask.png"

            img.save(image_file)
            m.save(mask_file)

            rows.append({
                "image_id": pair.image_id,
                "target_class": target_class,
                "x": x,
                "y": y,
                "tile_size": self.tile_size,
                "foreground_fraction": round(1.0 - frac if target_class == 0 else frac, 6),
                "classes_present": ",".join(map(str, vals)),
                "image_path": str(image_file),
                "mask_path": str(mask_file),
            })

        manifest = output_dir / "manifest.csv"

        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        wsi.close()

        return rows, manifest
