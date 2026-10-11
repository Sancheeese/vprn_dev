from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _numeric_suffix(path: Path) -> tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    return (int(matches[-1]) if matches else 10**12, path.name.lower())


class XRaySeries:
    """Read-only view of an X-ray DICOM directory."""

    def __init__(self, root: str | Path, read_fn: Callable[[str | Path], Any] | None = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"X-ray directory not found: {self.root}")
        self.files = sorted(
            (path for path in self.root.iterdir() if path.is_file() and path.suffix.lower() == ".dcm"),
            key=_numeric_suffix,
        )
        if not self.files:
            raise FileNotFoundError(f"no X-ray DICOM files found in: {self.root}")
        if read_fn is None:
            import pydicom

            read_fn = pydicom.dcmread
        self._read_fn = read_fn
        first = self._read_fn(self.files[0])
        self.metadata = {
            "DistanceSourceToDetector": first.get("DistanceSourceToDetector"),
            "IntensifierSize": first.get("IntensifierSize"),
        }

    def __len__(self) -> int:
        return len(self.files)

    def read_pixels(self, index: int) -> np.ndarray:
        return np.asarray(self._read_fn(self.files[index]).pixel_array)
