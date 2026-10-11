from __future__ import annotations

import inspect
from typing import Any

import torch


def split_model_output(output: Any) -> tuple[Any, Any | None]:
    """Normalize legacy model returns to ``(pose, optional_mask)``."""
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("model returned an empty output sequence")
        return output[0], output[1] if len(output) > 1 else None
    return output, None


def forward_model(model: Any, image: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[Any, Any | None]:
    """Call models with either the image-only or image-plus-mask legacy signature."""
    if mask is not None:
        try:
            signature = inspect.signature(model.forward)
        except (AttributeError, TypeError, ValueError):
            signature = inspect.signature(model)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        accepts_mask = (
            "mask" in signature.parameters
            or any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values())
            or len(positional) >= 2
        )
        if accepts_mask:
            return split_model_output(model(image, mask))
    return split_model_output(model(image))


def predict_pose(
    model: Any,
    image: torch.Tensor,
    dataset: Any,
    *,
    mask: torch.Tensor | None = None,
    device: str | torch.device | None = None,
) -> Any:
    """Run a pose regressor and lift its offset into the configured volume frame."""
    if device is not None:
        image = image.to(device)
        if mask is not None:
            mask = mask.to(device)
    model.eval()
    with torch.no_grad():
        offset, _predicted_mask = forward_model(model, image, mask)
    isocenter = getattr(dataset, "isocenter_pose", None)
    if isocenter is None or not hasattr(isocenter, "compose"):
        return offset
    return isocenter.to(offset.device).compose(offset)
