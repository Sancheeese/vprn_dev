from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def registration_loss(
    predicted_pose: Any,
    target_pose: Any,
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
    *,
    metric: Any,
    geodesic: Any,
    double_geodesic: Any,
    predicted_mask: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    mask_weight: float = 10.0,
) -> torch.Tensor:
    """Match the legacy NCC + SE(3) + mask objective used by case trainers."""
    ncc = metric(predicted_image, target_image)
    log_distance = geodesic(predicted_pose, target_pose)
    _rotation_distance, _translation_distance, double_distance = double_geodesic(
        predicted_pose, target_pose
    )
    loss = 1.0 - ncc + 1e-2 * (double_distance + log_distance)
    if predicted_mask is not None and target_mask is not None:
        if predicted_mask.shape != target_mask.shape:
            target_mask = F.interpolate(
                target_mask,
                size=predicted_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss = loss + mask_weight * torch.mean(
            (predicted_mask - target_mask) ** 2,
            dim=tuple(range(1, predicted_mask.ndim)),
        )
    return loss.mean()
