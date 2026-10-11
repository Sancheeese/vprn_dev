from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def _prepare_legacy_imports() -> None:
    project_root = Path(__file__).resolve().parents[4]
    for path in (project_root / "diffpose", project_root / "diffpose" / "ours"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_drr_class(*, bone: bool = False):
    _prepare_legacy_imports()
    module_name = "ours.utils.drr_bone_alter" if bone else "ours.utils.drr"
    return importlib.import_module(module_name).DRR


def create_drr(dataset: Any, *, height: int = 256, device: str = "cpu", bone: bool = False, **kwargs):
    drr_class = load_drr_class(bone=bone)
    params = {
        "volume": dataset.volume,
        "spacing": dataset.spacing,
        "sdr": dataset.sdr,
        "height": height,
        "delx": dataset.delx * 2,
        "reverse_x_axis": True,
    }
    params.update(kwargs)
    renderer = drr_class(**params).to(device)
    if bone and hasattr(renderer, "set_bone_attenuation_multiplier"):
        renderer.set_bone_attenuation_multiplier(
            float(kwargs.get("bone_attenuation_multiplier", 3.0))
        )
    return renderer
