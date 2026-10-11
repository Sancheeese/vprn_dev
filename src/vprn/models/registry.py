from __future__ import annotations

import importlib
import contextlib
import io
import sys
from pathlib import Path
from typing import Any

import torch


_MODEL_TARGETS = {
    "PoseRegressor": ("ours.utils.model2", "PoseRegressor"),
    "PoseRegressorAddSpDeco": ("ours.utils.model2", "PoseRegressorAddSpDeco"),
    "PoseRegressorCoeSpDeco": ("ours.utils.model2", "PoseRegressorCoeSpDeco"),
    "PoseRegressorGSpDeco": ("ours.utils.model2", "PoseRegressorGSpDeco"),
    "PoseRegressorSpAddDeco": ("ours.utils.model2", "PoseRegressorSpAddDeco"),
    "PoseRegressorCoe": ("ours.utils.model2", "PoseRegressorCoe"),
    "PoseRegressorCoeSp": ("ours.utils.model2", "PoseRegressorCoeSp"),
}


def supported_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_TARGETS))


def model_target(name: str) -> tuple[str, str]:
    try:
        return _MODEL_TARGETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {name}") from exc


def _prepare_legacy_imports() -> None:
    project_root = Path(__file__).resolve().parents[4]
    for path in (project_root / "diffpose", project_root / "diffpose" / "ours"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_model_class(name: str):
    module_name, class_name = model_target(name)
    _prepare_legacy_imports()
    return getattr(importlib.import_module(module_name), class_name)


def create_model(name: str, **model_params: Any):
    # Several legacy constructors print their entire backbone. Keep the
    # registry quiet so CLI logs contain experiment information only.
    with contextlib.redirect_stdout(io.StringIO()):
        return load_model_class(name)(**model_params)


def load_checkpoint(model: Any, checkpoint: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load either the legacy checkpoint envelope or a raw state dict."""

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
        metadata = _checkpoint_metadata(payload)
        metadata["missing_keys"] = list(incompatible.missing_keys)
        metadata["unexpected_keys"] = list(incompatible.unexpected_keys)
    elif isinstance(payload, dict):
        incompatible = model.load_state_dict(payload, strict=False)
        metadata = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    else:
        raise ValueError(f"unsupported checkpoint payload: {type(payload).__name__}")
    return metadata


def _checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    tensor_shapes: dict[str, list[int]] = {}
    for key, value in payload.items():
        if key == "model_state_dict":
            continue
        if isinstance(value, torch.Tensor):
            tensor_shapes[key] = list(value.shape)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
    if tensor_shapes:
        metadata["tensor_shapes"] = tensor_shapes
    return metadata
