from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class UncertaintyResult:
    mean_entropy: float
    mean_confidence: float
    disagreement: float
    entropy_map: np.ndarray


class UncertaintyAnalyzer:
    """
    Test-time augmentation uncertainty for semantic segmentation.

    Uses:
      1. original
      2. horizontal flip
      3. vertical flip
      4. horizontal + vertical flip

    Predictions are transformed back to the original orientation
    before averaging.
    """

    @staticmethod
    def _tta_inputs(x):
        return [
            x,
            torch.flip(x, dims=[3]),
            torch.flip(x, dims=[2]),
            torch.flip(x, dims=[2, 3]),
        ]

    @staticmethod
    def _invert_prediction(pred, index):
        if index == 0:
            return pred
        if index == 1:
            return torch.flip(pred, dims=[3])
        if index == 2:
            return torch.flip(pred, dims=[2])
        return torch.flip(pred, dims=[2, 3])

    @torch.no_grad()
    def analyze(self, model, images):
        tta_inputs = self._tta_inputs(images)

        probs = []

        for i, x in enumerate(tta_inputs):
            logits = model(x)
            p = torch.softmax(logits, dim=1)
            p = self._invert_prediction(p, i)
            probs.append(p)

        stacked = torch.stack(probs, dim=0)

        mean_prob = stacked.mean(dim=0)

        eps = 1e-8
        entropy = -(
            mean_prob * torch.log(mean_prob + eps)
        ).sum(dim=1)

        num_classes = mean_prob.shape[1]
        entropy = entropy / np.log(num_classes)

        confidence = mean_prob.max(dim=1).values

        predictions = stacked.argmax(dim=2)
        majority = mean_prob.argmax(dim=1)

        disagreement = (
            predictions != majority.unsqueeze(0)
        ).float().mean(dim=(0, 2, 3))

        return [
            UncertaintyResult(
                mean_entropy=float(entropy[i].mean().item()),
                mean_confidence=float(confidence[i].mean().item()),
                disagreement=float(disagreement[i].item()),
                entropy_map=entropy[i].cpu().numpy(),
            )
            for i in range(images.shape[0])
        ]
