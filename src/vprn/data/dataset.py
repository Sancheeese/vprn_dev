from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad

from ..config import CaseConfig
from .geometry import load_nifti_volume
from .xray import XRaySeries


def _rigid_transform_class():
    project_root = Path(__file__).resolve().parents[4]
    package_root = str(project_root / "diffpose")
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    return importlib.import_module("diffpose.calibration").RigidTransform


def _pose(rigid_transform, rotation: list[float], translation: np.ndarray | list[float]):
    return rigid_transform(
        torch.tensor([rotation], dtype=torch.float32),
        torch.as_tensor(np.asarray(translation), dtype=torch.float32).reshape(1, 3),
        "euler_angles",
        "ZYX",
    )


def _perspective_projection(extrinsic, intrinsic: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Project world points with the same camera convention as legacy datasets."""
    transformed = extrinsic.transform_points(points)
    projected = torch.einsum("ij,bnj->bni", intrinsic, transformed)
    return projected[..., :2] / projected[..., -1:].clone()


class UnifiedDataset(Dataset):
    """Common CT-X-ray Dataset with all legacy differences in CaseConfig."""

    def __init__(self, case: CaseConfig, *, preprocess: bool = True, xray_read_fn: Any = None):
        self.case = case
        self.preprocess = preprocess
        self.nii_path = case.ct_path
        self.xray = XRaySeries(case.xray_path, read_fn=xray_read_fn)
        self.volume, self.spacing, self.affine = load_nifti_volume(case.ct_path, case.geometry)
        self.x_file = [path.name for path in self.xray.files]
        self.x_filename = str(self.xray.files[0])

        distance = self.xray.metadata.get("DistanceSourceToDetector")
        intensifier = self.xray.metadata.get("IntensifierSize")
        if distance is None or intensifier is None:
            raise ValueError("X-ray metadata requires DistanceSourceToDetector and IntensifierSize")
        self.sdr = float(distance) / 2.0
        self.focal_len = float(distance)
        self.delx = float(intensifier) / 512.0
        self.intrinsic = torch.tensor(
            [[self.focal_len, 0.0, 0.0], [0.0, self.focal_len, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        rigid_transform = _rigid_transform_class()
        shape = np.asarray(self.volume.shape, dtype=np.float64)
        offset = np.asarray(
            [case.geometry.x_offset, case.geometry.y_offset, case.geometry.z_offset],
            dtype=np.float64,
        )
        isocenter_xyz = (shape - offset) * self.spacing / 2.0
        center_xyz = (shape - offset) * self.spacing / 2.0
        back_xyz = (-shape + offset) * self.spacing / 2.0
        self.isocenter_pose = _pose(rigid_transform, [-np.pi / 2, 0.0, np.pi / 2], isocenter_xyz)
        self.center_pose = _pose(rigid_transform, [0.0, 0.0, 0.0], center_xyz)
        self.back_pose = _pose(rigid_transform, [0.0, 0.0, 0.0], back_xyz)

        ras = self.affine @ np.array([0.0, 0.0, max(0, self.volume.shape[2] - 1), 1.0])
        origin = np.array([-ras[0], -ras[1], ras[2]], dtype=np.float32)
        origin[2] = -origin[2]
        origin = -origin
        self.lps2volume = rigid_transform(torch.eye(3), torch.tensor(origin))
        self.fiducials = self.get_fiducials()
        self.flip_xz = rigid_transform(
            torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.zeros(3),
        )
        self.translate = rigid_transform(
            torch.eye(3), torch.tensor([self.focal_len / 2.0, 0.0, 0.0])
        )
        self.flip_180 = rigid_transform(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
            torch.zeros(3),
        )

    def __len__(self) -> int:
        return len(self.xray)

    def __getitem__(self, index: int):
        if index == -1:
            return self.sample_synthetic()
        if index < -1:
            raise IndexError(f"dataset index out of range: {index}")
        image = torch.from_numpy(self.xray.read_pixels(index)).float().unsqueeze(0).unsqueeze(0)
        try:
            target = self.get_manual_gt(index)
        except FileNotFoundError:
            target = None
        return image, target

    def sample_synthetic(self, drr: Any | None = None, *, device: str | torch.device = "cpu"):
        """Render one legacy-style random training sample on demand."""
        if drr is None:
            from ..rendering.drr import create_drr

            drr = create_drr(self, height=256, device=device)
        from ..cli.legacy import random_offset

        isocenter = self.isocenter_pose.to(device)
        pose = isocenter.compose(self.back_pose.to(device)).compose(
            random_offset(1, device)
        ).compose(self.center_pose.to(device))
        return drr(None, None, None, pose=pose), pose

    def getInfo(self):
        return self.volume, self.spacing, self.sdr, self.delx, self.focal_len, self.lps2volume, self.intrinsic

    def get_x_filename(self, index: int) -> str:
        return self.x_file[index]

    def get_manual_gt(self, index: int | None = None):
        if index is None:
            index = 0
        if self.case.pose_mode == "legacy_zyl":
            if 0 <= index <= 10:
                values = self.case.pose_params["early"]
            elif 62 <= index <= 65:
                values = self.case.pose_params["late"]
            else:
                values = self.case.pose_params["default"]
            rigid_transform = _rigid_transform_class()
            return rigid_transform(
                torch.tensor([values[:3]], dtype=torch.float32),
                torch.tensor([values[3:]], dtype=torch.float32),
                parameterization="so3_log_map",
            )
        pose_path = self.case.gt_pose_path / f"pose_{index:04d}.json"
        if not pose_path.is_file():
            raise FileNotFoundError(f"pose file not found: {pose_path}")
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
        values = payload["pose_params"]
        rigid_transform = _rigid_transform_class()
        return rigid_transform(
            torch.tensor([values[:3]], dtype=torch.float32),
            torch.tensor([values[3:]], dtype=torch.float32),
            parameterization="so3_log_map",
        )

    def get_fiducials(self):
        if self.case.fiducial_path is None:
            return None
        payload = json.loads(self.case.fiducial_path.read_text(encoding="utf-8"))
        points = payload["markups"][0]["controlPoints"]
        coordinates = []
        for point in points:
            position = torch.tensor(point["position"], dtype=torch.float32)
            position[2] = -position[2]
            coordinates.append(position)
        if not coordinates:
            return torch.empty((1, 0, 3), dtype=torch.float32)
        return self.lps2volume.transform_points(torch.stack(coordinates).unsqueeze(0))

    def _project_fiducials(self, pose):
        device = pose.device
        flip_xz = self.flip_xz.to(device)
        translate = self.translate.to(device)
        total_pose = flip_xz.compose(translate).compose(pose.to(device))
        return _perspective_projection(
            total_pose.inverse(),
            self.intrinsic.to(device),
            self.fiducials.to(device),
        )

    def get_2d_fiducials(self, index: int, pose):
        if self.fiducials is None:
            raise ValueError("fiducial_path is required for 2D fiducial projection")
        true_fiducials = self._project_fiducials(self.get_manual_gt(index))
        predicted_fiducials = self._project_fiducials(pose)
        projection = self.case.fiducial_projection
        if projection == "swap_xy":
            true_fiducials = true_fiducials[..., [1, 0]] * torch.tensor([1.0, -1.0], device=true_fiducials.device)
            predicted_fiducials = predicted_fiducials[..., [1, 0]] * torch.tensor([1.0, -1.0], device=predicted_fiducials.device)
        elif projection == "negate_x":
            true_fiducials[..., 0] = -true_fiducials[..., 0]
            predicted_fiducials[..., 0] = -predicted_fiducials[..., 0]
        elif projection != "identity":
            raise ValueError(f"unknown fiducial projection: {projection}")
        return true_fiducials + 152.5, predicted_fiducials + 152.5

    def get_3d_fiducials(self, index: int, pose):
        if self.fiducials is None:
            raise ValueError("fiducial_path is required for 3D fiducial projection")
        def unproject(current_pose):
            points_2d = self._project_fiducials(current_pose)
            homogeneous = pad(points_2d, (0, 1), value=1)
            camera_points = self.focal_len * torch.einsum(
                "ij,bnj->bni", self.intrinsic.to(current_pose.device).inverse(), homogeneous
            )
            total_pose = self.flip_xz.to(current_pose.device).compose(
                self.translate.to(current_pose.device)
            ).compose(current_pose)
            return total_pose.transform_points(camera_points)
        return unproject(self.get_manual_gt(index)), unproject(pose)

    def calc_tre(self, index: int, pose):
        if self.fiducials is None:
            raise ValueError("fiducial_path is required for TRE")
        device = pose.device
        camera_frame = self.flip_xz.to(device).compose(self.translate.to(device))
        true_pose = camera_frame.compose(self.get_manual_gt(index).to(device)).inverse()
        predicted_pose = camera_frame.compose(pose.to(device)).inverse()
        fiducials = self.fiducials.to(device)
        true_points = true_pose.transform_points(fiducials)
        predicted_points = predicted_pose.transform_points(fiducials)
        return torch.norm(true_points - predicted_points, dim=-1).mean()

    def get_bone(self, index: int):
        import nibabel as nib

        def sort_key(path: Path):
            matches = re.findall(r"\d+", path.stem)
            return (int(matches[-1]) if matches else 10**12, path.name.lower())

        files = sorted(self.case.seg_path.rglob("*.nii*"), key=sort_key)
        if index >= len(files):
            raise IndexError(f"segmentation index {index} out of range ({len(files)} files)")
        data = np.asarray(nib.load(str(files[index])).dataobj, dtype=np.float32)
        tensor = torch.from_numpy(np.squeeze(data)).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return torch.nn.functional.interpolate(tensor, (256, 256), mode="bilinear", align_corners=False)
