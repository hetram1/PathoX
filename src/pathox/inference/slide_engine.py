from dataclasses import dataclass
from pathlib import Path
import csv
import json

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, TensorDataset

import openslide

from pathox.inference.uncertainty import UncertaintyAnalyzer
from pathox.models.unet import UNet


@dataclass
class SlideInferenceResult:
    image_id: str
    slide_dimensions: tuple[int, int]
    candidate_tiles: int
    analyzed_tiles: int
    tissue_fraction: float
    lesion_fraction: float
    mean_confidence: float
    mean_entropy: float
    mean_disagreement: float
    class_fractions: dict[int, float]


class SlideInferenceEngine:
    def __init__(
        self,
        checkpoint_path,
        device=None,
        tile_size=512,
        stride=512,
        thumbnail_size=2048,
        tissue_threshold=0.12,
        batch_size=4,
        uncertainty_top_k=50,
        num_classes=6,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tile_size = tile_size
        self.stride = stride
        self.thumbnail_size = thumbnail_size
        self.tissue_threshold = tissue_threshold
        self.batch_size = batch_size
        self.uncertainty_top_k = uncertainty_top_k
        self.num_classes = num_classes

        self.model = UNet(
            in_channels=3,
            num_classes=num_classes,
            base_channels=16,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.eval()

        self.uncertainty = UncertaintyAnalyzer()

    @staticmethod
    def _tissue_mask(thumbnail):
        rgb = np.asarray(thumbnail.convert("RGB"))

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV,
        )

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # White background has low saturation and high brightness.
        mask = (
            (saturation > 18)
            & (value < 250)
        ).astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return mask > 0

    def _candidate_tiles(self, slide, tissue_mask):
        thumb_h, thumb_w = tissue_mask.shape
        slide_w, slide_h = slide.dimensions

        scale_x = thumb_w / slide_w
        scale_y = thumb_h / slide_h

        candidates = []

        for y in range(
            0,
            slide_h - self.tile_size + 1,
            self.stride,
        ):
            for x in range(
                0,
                slide_w - self.tile_size + 1,
                self.stride,
            ):
                tx0 = int(x * scale_x)
                ty0 = int(y * scale_y)

                tx1 = max(
                    tx0 + 1,
                    int((x + self.tile_size) * scale_x),
                )
                ty1 = max(
                    ty0 + 1,
                    int((y + self.tile_size) * scale_y),
                )

                region = tissue_mask[
                    ty0:min(ty1, thumb_h),
                    tx0:min(tx1, thumb_w),
                ]

                fraction = (
                    float(region.mean())
                    if region.size
                    else 0.0
                )

                if fraction >= self.tissue_threshold:
                    candidates.append(
                        {
                            "x": x,
                            "y": y,
                            "tissue_fraction": fraction,
                        }
                    )

        return candidates

    @staticmethod
    def _to_tensor(pil_image):
        arr = np.array(
            pil_image.convert("RGB"),
            dtype=np.uint8,
            copy=True,
        )

        tensor = (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        return tensor

    def _read_batch(self, slide, records):
        tensors = []

        for record in records:
            image = slide.read_region(
                (record["x"], record["y"]),
                0,
                (
                    self.tile_size,
                    self.tile_size,
                ),
            ).convert("RGB")

            tensors.append(
                self._to_tensor(image)
            )

        return torch.stack(tensors)

    @staticmethod
    def _entropy(probs):
        eps = 1e-8
        entropy = -(
            probs * torch.log(probs + eps)
        ).sum(dim=1)

        entropy /= np.log(probs.shape[1])

        return entropy

    @staticmethod
    def _save_heatmap(
        values,
        records,
        slide,
        output_path,
        title,
        cmap="Reds",
    ):
        thumb = slide.get_thumbnail(
            (2048, 2048)
        ).convert("RGB")

        canvas = np.asarray(thumb).copy()

        slide_w, slide_h = slide.dimensions
        thumb_w, thumb_h = thumb.size

        scale_x = thumb_w / slide_w
        scale_y = thumb_h / slide_h

        overlay = np.zeros(
            (thumb_h, thumb_w),
            dtype=np.float32,
        )

        counts = np.zeros_like(overlay)

        for value, record in zip(values, records):
            x0 = int(record["x"] * scale_x)
            y0 = int(record["y"] * scale_y)

            x1 = max(
                x0 + 1,
                int(
                    (record["x"] + self_tile_size)
                    * scale_x
                ),
            )
            y1 = max(
                y0 + 1,
                int(
                    (record["y"] + self_tile_size)
                    * scale_y
                ),
            )

            x1 = min(x1, thumb_w)
            y1 = min(y1, thumb_h)

            overlay[y0:y1, x0:x1] += value
            counts[y0:y1, x0:x1] += 1.0

        counts[counts == 0] = 1
        overlay /= counts

        norm = np.clip(
            overlay,
            0,
            1,
        )

        heat = cv2.applyColorMap(
            (norm * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        heat = cv2.cvtColor(
            heat,
            cv2.COLOR_BGR2RGB,
        )

        fused = (
            0.60 * canvas +
            0.40 * heat
        ).astype(np.uint8)

        result = Image.fromarray(fused)

        draw = ImageDraw.Draw(result)
        draw.rectangle(
            (0, 0, 220, 34),
            fill="black",
        )
        draw.text(
            (8, 8),
            title,
            fill="white",
        )

        result.save(output_path)

    def infer(
        self,
        slide_path,
        output_dir,
        image_id=None,
        save_hard_tiles=True,
    ):
        slide_path = Path(slide_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if image_id is None:
            image_id = slide_path.stem

        slide = openslide.OpenSlide(
            str(slide_path)
        )

        thumbnail = slide.get_thumbnail(
            (
                self.thumbnail_size,
                self.thumbnail_size,
            )
        )

        tissue = self._tissue_mask(
            thumbnail
        )

        tissue_fraction = float(
            tissue.mean()
        )

        tissue.save if False else None

        candidates = self._candidate_tiles(
            slide,
            tissue,
        )

        if not candidates:
            slide.close()
            raise RuntimeError(
                "No tissue-containing tiles found"
            )

        predictions = []
        confidences = []
        entropies = []
        lesion_fractions = []

        # Standard inference over every tissue tile.
        for start in range(
            0,
            len(candidates),
            self.batch_size,
        ):
            batch_records = candidates[
                start:start + self.batch_size
            ]

            batch = self._read_batch(
                slide,
                batch_records,
            ).to(
                self.device,
                non_blocking=True,
            )

            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.softmax(
                    logits,
                    dim=1,
                )
                pred = torch.argmax(
                    probs,
                    dim=1,
                )

                entropy = self._entropy(probs)
                confidence = probs.max(
                    dim=1
                ).values

            for i in range(
                len(batch_records)
            ):
                p = pred[i].cpu().numpy()

                predictions.append(p)

                confidences.append(
                    float(
                        confidence[i].mean().item()
                    )
                )

                entropies.append(
                    float(
                        entropy[i].mean().item()
                    )
                )

                lesion_fractions.append(
                    float(
                        (p > 0).mean()
                    )
                )

        # Run expensive TTA uncertainty only on the
        # most ambiguous standard predictions.
        hard_indices = np.argsort(
            np.asarray(entropies)
        )[::-1][
            :min(
                self.uncertainty_top_k,
                len(candidates),
            )
        ]

        final_entropy = list(
            entropies
        )
        final_confidence = list(
            confidences
        )
        disagreement = np.zeros(
            len(candidates),
            dtype=np.float32,
        )

        for start in range(
            0,
            len(hard_indices),
            self.batch_size,
        ):
            indices = hard_indices[
                start:start + self.batch_size
            ]

            records = [
                candidates[int(i)]
                for i in indices
            ]

            batch = self._read_batch(
                slide,
                records,
            ).to(self.device)

            results = self.uncertainty.analyze(
                self.model,
                batch,
            )

            for local, global_idx in enumerate(
                indices
            ):
                r = results[local]

                final_entropy[
                    int(global_idx)
                ] = r.mean_entropy

                final_confidence[
                    int(global_idx)
                ] = r.mean_confidence

                disagreement[
                    int(global_idx)
                ] = r.disagreement

        # Pixel-level class totals across analyzed tissue tiles.
        class_counts = np.zeros(
            self.num_classes,
            dtype=np.int64,
        )

        for pred in predictions:
            class_counts += np.bincount(
                pred.reshape(-1),
                minlength=self.num_classes,
            )

        total_pixels = int(
            class_counts.sum()
        )

        class_fractions = {
            c: float(
                class_counts[c] /
                total_pixels
            )
            for c in range(
                self.num_classes
            )
        }

        result = SlideInferenceResult(
            image_id=image_id,
            slide_dimensions=slide.dimensions,
            candidate_tiles=len(candidates),
            analyzed_tiles=len(candidates),
            tissue_fraction=tissue_fraction,
            lesion_fraction=float(
                np.mean(lesion_fractions)
            ),
            mean_confidence=float(
                np.mean(final_confidence)
            ),
            mean_entropy=float(
                np.mean(final_entropy)
            ),
            mean_disagreement=float(
                np.mean(disagreement)
            ),
            class_fractions=class_fractions,
        )

        # Tile-level CSV.
        tile_csv = output_dir / "tile_predictions.csv"

        with tile_csv.open(
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "x",
                    "y",
                    "tissue_fraction",
                    "lesion_fraction",
                    "mean_confidence",
                    "mean_entropy",
                    "disagreement",
                ]
            )

            for i, record in enumerate(
                candidates
            ):
                writer.writerow(
                    [
                        record["x"],
                        record["y"],
                        record["tissue_fraction"],
                        lesion_fractions[i],
                        final_confidence[i],
                        final_entropy[i],
                        float(disagreement[i]),
                    ]
                )

        # Hard-tile table.
        ranking = sorted(
            range(len(candidates)),
            key=lambda i: (
                final_entropy[i],
                disagreement[i],
            ),
            reverse=True,
        )

        hard_csv = output_dir / "hard_tiles.csv"

        with hard_csv.open(
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "x",
                    "y",
                    "entropy",
                    "confidence",
                    "disagreement",
                    "lesion_fraction",
                ]
            )

            for rank, i in enumerate(
                ranking[:self.uncertainty_top_k],
                start=1,
            ):
                writer.writerow(
                    [
                        rank,
                        candidates[i]["x"],
                        candidates[i]["y"],
                        final_entropy[i],
                        final_confidence[i],
                        float(disagreement[i]),
                        lesion_fractions[i],
                    ]
                )

        if save_hard_tiles:
            hard_dir = output_dir / "hard_tiles"
            hard_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for rank, i in enumerate(
                ranking[:10],
                start=1,
            ):
                image = slide.read_region(
                    (
                        candidates[i]["x"],
                        candidates[i]["y"],
                    ),
                    0,
                    (
                        self.tile_size,
                        self.tile_size,
                    ),
                ).convert("RGB")

                image.save(
                    hard_dir /
                    f"rank_{rank:02d}_"
                    f"x{candidates[i]['x']}_"
                    f"y{candidates[i]['y']}.png"
                )

        # Heatmaps.
        global self_tile_size
        self_tile_size = self.tile_size

        lesion_heatmap = output_dir / "lesion_heatmap.png"
        self._save_heatmap(
            lesion_fractions,
            candidates,
            slide,
            lesion_heatmap,
            "PathoX Lesion Probability",
        )

        uncertainty_heatmap = (
            output_dir /
            "uncertainty_heatmap.png"
        )

        self._save_heatmap(
            final_entropy,
            candidates,
            slide,
            uncertainty_heatmap,
            "PathoX Uncertainty",
        )

        # Overview with candidate-tile locations.
        overview = thumbnail.copy()
        draw = ImageDraw.Draw(
            overview
        )

        thumb_w, thumb_h = overview.size
        slide_w, slide_h = slide.dimensions

        sx = thumb_w / slide_w
        sy = thumb_h / slide_h

        for i, record in enumerate(
            candidates
        ):
            x0 = int(record["x"] * sx)
            y0 = int(record["y"] * sy)
            x1 = int(
                (record["x"] + self.tile_size) * sx
            )
            y1 = int(
                (record["y"] + self.tile_size) * sy
            )

            if i in set(
                ranking[:10]
            ):
                draw.rectangle(
                    (x0, y0, x1, y1),
                    outline="red",
                    width=3,
                )

        overview.save(
            output_dir / "slide_overview.png"
        )

        report = {
            "image_id": result.image_id,
            "slide_dimensions": list(
                result.slide_dimensions
            ),
            "candidate_tiles": result.candidate_tiles,
            "analyzed_tiles": result.analyzed_tiles,
            "tissue_fraction_thumbnail": result.tissue_fraction,
            "mean_predicted_lesion_fraction": result.lesion_fraction,
            "mean_confidence": result.mean_confidence,
            "mean_entropy": result.mean_entropy,
            "mean_disagreement": result.mean_disagreement,
            "class_fractions": {
                str(k): v
                for k, v in result.class_fractions.items()
            },
            "model": {
                "tile_size": self.tile_size,
                "stride": self.stride,
                "tissue_threshold": self.tissue_threshold,
                "uncertainty_top_k": self.uncertainty_top_k,
            },
        }

        with (
            output_dir / "slide_report.json"
        ).open("w") as f:
            json.dump(
                report,
                f,
                indent=2,
            )

        slide.close()

        return result
