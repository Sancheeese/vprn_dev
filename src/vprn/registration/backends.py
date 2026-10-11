from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch


def _legacy_sparse_registration():
    project_root = Path(__file__).resolve().parents[4]
    for path in (project_root / "diffpose", project_root / "diffpose" / "ours"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module("ours.utils.registration_bone").SparseRegistration


def sparse_refine(
    initial_pose: Any,
    target_image: torch.Tensor,
    drr: Any,
    drr_bone: Any,
    *,
    parameterization: str = "se3_log_map",
    convention: str | None = None,
    iterations: int = 50,
    learning_rate_rotation: float = 15e-3,
    learning_rate_translation: float = 7.5,
    rotation_lr: float | None = None,
    translation_lr: float | None = None,
    optimizer: str = "adam",
    maximize: bool = True,
) -> Any:
    """Refine a pose with the legacy differentiable sparse renderer."""

    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if iterations == 0:
        return initial_pose
    sparse_registration = _legacy_sparse_registration()
    registration = sparse_registration(
        drr,
        drr_bone,
        pose=initial_pose,
        parameterization=parameterization,
        convention=convention,
    )
    from .optimizers import create_pose_optimizer

    optimizer = create_pose_optimizer(
        registration,
        rotation_lr=learning_rate_rotation if rotation_lr is None else float(rotation_lr),
        translation_lr=learning_rate_translation if translation_lr is None else float(translation_lr),
        optimizer=optimizer,
        maximize=maximize,
    )
    metric_cls = importlib.import_module("diffdrr.metrics").MultiscaleNormalizedCrossCorrelation2d
    metric = metric_cls(eps=1e-4)
    for _ in range(iterations):
        predicted, _predicted_bone, _mask, _pose = registration()
        score = metric(predicted, target_image).mean()
        optimizer.zero_grad(set_to_none=True)
        score.backward()
        optimizer.step()
    return registration.get_current_pose()


def cmaes_refine(
    initial_pose: Any,
    target_image: torch.Tensor,
    drr: Any,
    *,
    iterations: int = 50,
    sigma0: float = 1.0,
    population_size: int | None = None,
    parameterization: str = "se3_log_map",
    convention: str | None = None,
) -> Any:
    """Refine a pose with CMA-ES against the differentiable renderer output."""
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if iterations == 0:
        return initial_pose
    try:
        import cma
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("cmaes refinement requires the 'cma' package") from exc
    if not hasattr(initial_pose, "get_rotation") or not hasattr(initial_pose, "get_translation"):
        raise TypeError("CMA-ES refinement requires a RigidTransform-like initial pose")
    from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d

    del parameterization, convention
    rotation = initial_pose.get_rotation("so3_log_map").detach().cpu().reshape(-1)
    translation = initial_pose.get_translation().detach().cpu().reshape(-1)
    x0 = torch.cat([rotation, translation]).tolist()
    target = target_image.detach()
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)

    def objective(values: list[float]) -> float:
        params = torch.tensor(values, dtype=torch.float32, device=target.device)
        pose_type = type(initial_pose)
        pose = pose_type(
            params[:3].reshape(1, 3).to(target.device),
            params[3:].reshape(1, 3).to(target.device),
            parameterization="so3_log_map",
        )
        with torch.no_grad():
            rendered = drr(None, None, None, pose=pose)
            while rendered.ndim < target.ndim:
                rendered = rendered.unsqueeze(0)
            score = metric(rendered, target).mean()
        return float((1.0 - score).detach().cpu())

    options: dict[str, Any] = {"maxiter": int(iterations), "verb_disp": 0}
    if population_size is not None:
        options["popsize"] = int(population_size)
    result = cma.fmin(objective, x0, float(sigma0), options)
    best = torch.tensor(result[0], dtype=torch.float32, device=target.device)
    return type(initial_pose)(
        best[:3].reshape(1, 3),
        best[3:].reshape(1, 3),
        parameterization="so3_log_map",
    )


def refine_pose(
    initial_pose: Any,
    target_image: torch.Tensor,
    drr: Any,
    drr_bone: Any | None = None,
    *,
    method: str = "sparse",
    **kwargs: Any,
) -> Any:
    """Dispatch the configured pose refinement backend.

    The legacy ``gradient`` and ``sparse`` registrations both optimize the
    differentiable sparse renderer; keeping both names preserves old CLI
    invocations while sharing one implementation.
    """
    iterations = int(kwargs.get("iterations", 50))
    if iterations == 0:
        return initial_pose
    if method in {"sparse", "gradient"}:
        if drr_bone is None:
            raise ValueError(f"{method} refinement requires drr_bone")
        return sparse_refine(initial_pose, target_image, drr, drr_bone, **kwargs)
    if method == "cmaes":
        return cmaes_refine(initial_pose, target_image, drr, **kwargs)
    if method == "pose_regression":
        return initial_pose
    raise ValueError(f"unknown registration method: {method}")
