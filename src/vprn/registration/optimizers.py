from __future__ import annotations

from typing import Any

import torch


def create_pose_optimizer(
    registration: Any,
    *,
    rotation_lr: float = 15e-3,
    translation_lr: float = 7.5,
    optimizer: str = "adam",
    maximize: bool = True,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Build the legacy two-rate optimizer used by sparse/gradient registration."""
    if rotation_lr <= 0 or translation_lr <= 0:
        raise ValueError("rotation_lr and translation_lr must be positive")
    params = [
        {"params": [registration.rotation], "lr": rotation_lr},
        {"params": [registration.translation], "lr": translation_lr},
    ]
    name = optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, maximize=maximize)
    if name == "sgd":
        return torch.optim.SGD(params, maximize=maximize, momentum=momentum)
    raise ValueError(f"unknown pose optimizer: {optimizer}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    kind: str = "step",
    step_size: int = 25,
    gamma: float = 0.9,
    total_steps: int | None = None,
    warmup_steps: int = 0,
) -> Any | None:
    """Create a scheduler without tying callers to one legacy dependency."""
    name = kind.lower()
    if name in {"none", "constant"}:
        return None
    if name == "step":
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        if total_steps is None or total_steps <= 0:
            raise ValueError("total_steps is required for cosine scheduling")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if name == "warmup_cosine":
        if total_steps is None or total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps")
        def factor(step: int) -> float:
            if step < warmup_steps:
                return max(1e-8, (step + 1) / max(1, warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))).item()
        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    raise ValueError(f"unknown scheduler: {kind}")
