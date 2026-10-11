from __future__ import annotations

import importlib
import sys
from pathlib import Path


def random_offset(batch_size: int, device):
    project_root = Path(__file__).resolve().parents[4]
    ours_root = str(project_root / "diffpose" / "ours")
    if ours_root not in sys.path:
        sys.path.insert(0, ours_root)
    return importlib.import_module("ours.utils.CT_dataset").get_random_offset(batch_size, device)
