from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class GeometryConfig:
    """Per-case geometry choices retained from the legacy pipeline."""

    x_offset: int = 0
    y_offset: int = 0
    z_offset: int = 0
    z_cut: int = 0
    z_cut_end: int = -1
    factors: tuple[float, float, float] = (1.0, 1.0, 1.0)
    axis_transform: str = "identity"
    spacing_permutation: tuple[int, int, int] | None = None
    post_flip_axes: tuple[int, ...] = (0,)
    isocenter_mode: str = "legacy_center"


@dataclass(frozen=True)
class CaseConfig:
    """All inputs and choices required to run one clinical case."""

    case_id: str
    patient_name: str
    ct_series: str
    ct_path: Path
    xray_path: Path
    seg_path: Path
    gt_pose_path: Path
    fiducial_path: Path | None
    model: str
    checkpoint: Path
    geometry: GeometryConfig
    model_params: dict[str, Any] = field(default_factory=dict)
    registration_method: str = "pose_regression"
    registration_params: dict[str, Any] = field(default_factory=dict)
    training_params: dict[str, Any] = field(default_factory=dict)
    pose_mode: str = "json"
    pose_params: dict[str, Any] = field(default_factory=dict)
    fiducial_projection: str = "negate_x"
    train_model: str | None = None
    train_checkpoint: Path | None = None
    register_model: str | None = None
    register_checkpoint: Path | None = None


def _project_root(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    # cases.json is expected at <project>/configs/cases.json.
    return resolved.parent.parent


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _geometry_from_mapping(raw: Mapping[str, Any]) -> GeometryConfig:
    factors = tuple(float(value) for value in raw.get("factors", (1.0, 1.0, 1.0)))
    if len(factors) != 3:
        raise ValueError("geometry.factors must contain exactly three values")
    post_flip_axes = tuple(int(value) for value in raw.get("post_flip_axes", (0,)))
    if any(axis not in (0, 1, 2) for axis in post_flip_axes):
        raise ValueError("geometry.post_flip_axes values must be 0, 1, or 2")
    spacing_permutation = None
    if raw.get("spacing_permutation") is not None:
        spacing_permutation = tuple(int(value) for value in raw["spacing_permutation"])
        if len(spacing_permutation) != 3 or sorted(spacing_permutation) != [0, 1, 2]:
            raise ValueError("geometry.spacing_permutation must be a permutation of [0, 1, 2]")
    return GeometryConfig(
        x_offset=int(raw.get("x_offset", 0)),
        y_offset=int(raw.get("y_offset", 0)),
        z_offset=int(raw.get("z_offset", 0)),
        z_cut=int(raw.get("z_cut", 0)),
        z_cut_end=int(raw.get("z_cut_end", -1)),
        factors=factors,
        axis_transform=str(raw.get("axis_transform", "identity")),
        spacing_permutation=spacing_permutation,
        post_flip_axes=post_flip_axes,
        isocenter_mode=str(raw.get("isocenter_mode", "legacy_center")),
    )


def load_cases(config_path: str | Path) -> dict[str, CaseConfig]:
    config_path = Path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Mapping):
        raise ValueError("configuration must contain a 'cases' object")

    project_root = _project_root(config_path)
    cases: dict[str, CaseConfig] = {}
    for case_id, raw in raw_cases.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"case {case_id!r} must be an object")
        try:
            registration_method = str(raw.get("registration_method", "pose_regression"))
            if registration_method not in {"pose_regression", "sparse", "gradient", "cmaes"}:
                raise ValueError(
                    f"case {case_id!r} has unsupported registration_method {registration_method!r}"
                )
            fiducial_projection = str(raw.get("fiducial_projection", "negate_x"))
            if fiducial_projection not in {"identity", "negate_x", "swap_xy"}:
                raise ValueError(
                    f"case {case_id!r} has unsupported fiducial_projection {fiducial_projection!r}"
                )
            cases[case_id] = CaseConfig(
                case_id=str(case_id),
                patient_name=str(raw["patient_name"]),
                ct_series=str(raw["ct_series"]),
                ct_path=_resolve_path(raw["ct_path"], project_root),
                xray_path=_resolve_path(raw["xray_path"], project_root),
                seg_path=_resolve_path(raw["seg_path"], project_root),
                gt_pose_path=_resolve_path(raw["gt_pose_path"], project_root),
                fiducial_path=(
                    _resolve_path(raw["fiducial_path"], project_root)
                    if raw.get("fiducial_path")
                    else None
                ),
                model=str(raw["model"]),
                checkpoint=_resolve_path(raw["checkpoint"], project_root),
                geometry=_geometry_from_mapping(raw.get("geometry", {})),
                model_params=dict(raw.get("model_params", {})),
                registration_method=registration_method,
                registration_params=dict(raw.get("registration_params", {})),
                training_params=dict(raw.get("training_params", {})),
                pose_mode=str(raw.get("pose_mode", "json")),
                pose_params=dict(raw.get("pose_params", {})),
                fiducial_projection=fiducial_projection,
                train_model=str(raw["train_model"]) if raw.get("train_model") else None,
                train_checkpoint=(
                    _resolve_path(raw["train_checkpoint"], project_root)
                    if raw.get("train_checkpoint")
                    else None
                ),
                register_model=str(raw["register_model"]) if raw.get("register_model") else None,
                register_checkpoint=(
                    _resolve_path(raw["register_checkpoint"], project_root)
                    if raw.get("register_checkpoint")
                    else None
                ),
            )
        except KeyError as exc:
            raise ValueError(f"case {case_id!r} is missing required field {exc.args[0]!r}") from exc
    return cases


def load_case(case_id: str, config_path: str | Path) -> CaseConfig:
    cases = load_cases(config_path)
    try:
        return cases[case_id]
    except KeyError as exc:
        raise KeyError(f"Unknown case: {case_id}") from exc
