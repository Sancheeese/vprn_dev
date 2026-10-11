from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        return tensor.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class ArtifactStore:
    """Stable, isolated output paths for one case and one experiment run."""

    def __init__(
        self,
        runs_root: str | Path,
        case_id: str,
        run_id: str,
        *,
        include_checkpoints: bool = True,
    ):
        self.root = Path(runs_root) / case_id / run_id
        self.registration_dir = self.root / "registration"
        self.metrics_dir = self.root / "metrics"
        self.figures_dir = self.root / "figures"
        self.checkpoints_dir = self.root / "checkpoints"
        self.logs_dir = self.root / "logs"
        self.include_checkpoints = include_checkpoints

    def ensure(self) -> None:
        paths = [self.registration_dir, self.metrics_dir, self.figures_dir, self.logs_dir]
        if self.include_checkpoints:
            paths.append(self.checkpoints_dir)
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def _path(self, relative_path: str | Path) -> Path:
        path = self.root / relative_path
        if path.is_absolute() and self.root not in path.parents:
            raise ValueError("artifact path must stay inside the run directory")
        return path

    def save_config(self, config: Mapping[str, Any]) -> Path:
        return self.write_json("config.json", dict(config))

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_csv(self, relative_path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(rows)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
