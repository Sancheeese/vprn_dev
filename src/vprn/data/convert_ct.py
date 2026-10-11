from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class ConversionResult:
    status: str
    destination: Path
    provenance_path: Path
    selected_count: int = 0
    dropped_count: int = 0
    reason: str | None = None


def _load_mrct_converter() -> ModuleType:
    project_root = Path(__file__).resolve().parents[4]
    source_path = project_root / "mrct" / "convert_dicom_to_nii.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"mrct converter not found: {source_path}")

    spec = importlib.util.spec_from_file_location("vprn_mrct_converter", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load mrct converter: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _provenance_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.json")


def _write_provenance(
    path: Path,
    *,
    source_dir: Path,
    destination: Path,
    patient_name: str,
    series_name: str,
    result: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "patient_name": patient_name,
        "modality": "CT",
        "series_name": series_name,
        "source_series_dir": str(source_dir),
        "destination": str(destination),
        "selected_count": result.get("selected_count", 0),
        "dropped_count": result.get("dropped_count", 0),
        "converter_status": result.get("status"),
        "converter_reason": result.get("reason"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_ct_series(
    source_dir: str | Path,
    destination: str | Path,
    patient_name: str,
    series_name: str,
    *,
    overwrite: bool = False,
    min_dicom_files: int = 10,
    converter: ModuleType | Any | None = None,
) -> ConversionResult:
    """Convert one explicitly selected CT DICOM series to NIfTI.

    The converter is deliberately constrained to modality ``CT``. X-ray and
    segmentation paths are outside this function and are never scanned.
    """

    source_dir = Path(source_dir)
    destination = Path(destination)
    provenance_path = _provenance_path(destination)
    if destination.exists() and not overwrite:
        return ConversionResult("skipped", destination, provenance_path, reason="destination exists")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"CT series directory not found: {source_dir}")

    converter = converter or _load_mrct_converter()
    study = converter.Study(patient_name, "CT", source_dir)
    with tempfile.TemporaryDirectory(prefix="diffpose_ct_conversion_") as staging_dir:
        result = converter.convert_study(
            study,
            Path(staging_dir),
            overwrite=True,
            min_dicom_files=min_dicom_files,
        )
        if result.get("status") != "converted":
            return ConversionResult(
                str(result.get("status", "skipped")),
                destination,
                provenance_path,
                selected_count=int(result.get("selected_count", 0)),
                dropped_count=int(result.get("dropped_count", 0)),
                reason=result.get("reason"),
            )

        generated_path = Path(result["nii_path"])
        if not generated_path.is_file():
            raise FileNotFoundError(f"converter did not create NIfTI: {generated_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_path, destination)

    _write_provenance(
        provenance_path,
        source_dir=source_dir,
        destination=destination,
        patient_name=patient_name,
        series_name=series_name,
        result=result,
    )
    return ConversionResult(
        "converted",
        destination,
        provenance_path,
        selected_count=int(result.get("selected_count", 0)),
        dropped_count=int(result.get("dropped_count", 0)),
    )
