from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .initialization import forward_model


@dataclass(frozen=True)
class RegistrationResult:
    index: int
    prediction: Any
    target: Any
    predicted_mask: Any | None = None


class RegistrationEngine:
    """Common registration orchestration around a model and Dataset.

    A richer optimizer backend can be supplied later without changing the
    Dataset/model/CLI contract.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        *,
        device: str | torch.device = "cpu",
        preprocess: Any | None = None,
        mask_preprocess: Any | None = None,
    ):
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device)
        self.preprocess = preprocess
        self.mask_preprocess = mask_preprocess

    def run(self, index: int) -> RegistrationResult:
        image, target = self.dataset[index]
        image = image.to(self.device) if isinstance(image, torch.Tensor) else image
        if self.preprocess is not None:
            image = self.preprocess(image)
        mask = None
        get_bone = getattr(self.dataset, "get_bone", None)
        if callable(get_bone):
            try:
                mask = get_bone(index)
                if isinstance(mask, torch.Tensor):
                    mask = mask.to(self.device)
                if self.mask_preprocess is not None:
                    mask = self.mask_preprocess(mask)
            except (FileNotFoundError, IndexError):
                mask = None
        prediction, predicted_mask = forward_model(self.model, image, mask)
        if hasattr(prediction, "get_matrix") and hasattr(self.dataset, "isocenter_pose"):
            prediction = self.dataset.isocenter_pose.to(self.device).compose(prediction)
        if isinstance(prediction, torch.Tensor):
            prediction = float(prediction.detach().mean().cpu())
        return RegistrationResult(
            index=index,
            prediction=prediction,
            target=target,
            predicted_mask=predicted_mask,
        )
