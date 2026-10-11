from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import CaseConfig
from ..data.dataset import UnifiedDataset
from ..models.registry import create_model


def checkpoint_run_dir(checkpoints_root: str | Path, case_id: str, run_id: str) -> Path:
    """Return the case/run directory used for trained model artifacts."""
    return Path(checkpoints_root) / case_id / run_id


def default_model_params(case: CaseConfig) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model_name": "resnet18",
        "parameterization": "se3_log_map",
        "convention": None,
        "pretrained": False,
    }
    params.update(case.model_params)
    return params


def build_dataset_and_model(case: CaseConfig, device: str, *, role: str = "register"):
    dataset = build_dataset(case)
    model = build_model(case, device, role=role)
    return dataset, model


def build_dataset(case: CaseConfig) -> UnifiedDataset:
    return UnifiedDataset(case)


def build_model(case: CaseConfig, device: str, *, role: str = "register"):
    if role == "train":
        model_name = case.train_model or case.model
    else:
        model_name = case.register_model or case.model
    return create_model(model_name, **default_model_params(case)).to(device)


def case_snapshot(case: CaseConfig, *, role: str, **extra: Any) -> dict[str, Any]:
    """Return a serializable, complete config snapshot for an artifact run."""
    from dataclasses import asdict

    snapshot = asdict(case)
    snapshot["role"] = role
    snapshot["model"] = (
        (case.train_model or case.model)
        if role == "train"
        else (case.register_model or case.model)
    )
    snapshot.update(extra)
    return snapshot


def model_prediction_record(index: int, prediction: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"index": index}
    if hasattr(prediction, "get_rotation") and hasattr(prediction, "get_translation"):
        rotation = prediction.get_rotation("euler_angles", "ZYX").detach().cpu().reshape(-1).tolist()
        translation = prediction.get_translation().detach().cpu().reshape(-1).tolist()
        record.update({"rx": rotation[0], "ry": rotation[1], "rz": rotation[2]})
        record.update({"tx": translation[0], "ty": translation[1], "tz": translation[2]})
    elif isinstance(prediction, torch.Tensor):
        values = prediction.detach().cpu().reshape(-1).tolist()
        record.update({f"value_{i}": value for i, value in enumerate(values)})
    else:
        record["value"] = prediction
    return record
