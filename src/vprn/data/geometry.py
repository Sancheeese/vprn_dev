from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from ..config import GeometryConfig


_PERMUTATIONS = {
    "transpose_0_1": (1, 0, 2),
    "transpose_0_2": (2, 1, 0),
    "transpose_1_2": (0, 2, 1),
    "ysy_legacy": (0, 2, 1),
    # These NIfTI files are written in SimpleITK's (x, y, z) order while
    # the legacy DICOM loaders use different explicit swapaxes sequences.
    "xyl_legacy": (0, 2, 1),
    "zyl_legacy": (2, 0, 1),
}


def _apply_axis_transform(data: np.ndarray, name: str) -> tuple[np.ndarray, tuple[int, int, int]]:
    if name == "identity":
        return data, (0, 1, 2)
    if name in {"sxh_legacy", "wfl_legacy"}:
        return np.flip(data, axis=2), (0, 1, 2)
    if name in _PERMUTATIONS:
        permutation = _PERMUTATIONS[name]
        return np.transpose(data, permutation), permutation
    raise ValueError(f"unknown axis transform: {name}")


def load_nifti_volume(
    path: str | Path,
    geometry: GeometryConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a CT NIfTI and apply the configured legacy voxel pipeline."""

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    if data.ndim != 3:
        raise ValueError(f"CT NIfTI must be 3D, got shape {data.shape}")
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    data, permutation = _apply_axis_transform(data, geometry.axis_transform)
    spacing = spacing[list(geometry.spacing_permutation or permutation)]

    if geometry.z_cut_end != -1:
        data = data[:, :, max(0, geometry.z_cut) : geometry.z_cut_end]
    elif geometry.z_cut > 0:
        data = data[:, :, : geometry.z_cut]

    factors = np.asarray(geometry.factors, dtype=np.float64)
    if np.any(factors <= 0):
        raise ValueError("geometry.factors values must be positive")
    if not np.allclose(factors, 1.0):
        data = zoom(data, tuple(float(value) for value in factors), order=1)
        spacing = spacing / factors

    for axis in geometry.post_flip_axes:
        data = np.flip(data, axis=axis)
    return np.ascontiguousarray(data), spacing, np.asarray(image.affine, dtype=np.float64)
