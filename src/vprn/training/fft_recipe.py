"""Shared input recipe matching the original ysy/sxh FFT trainers."""
from __future__ import annotations

import json
import random
from dataclasses import replace

import numpy as np
import torch

from ..cli.common import build_dataset
from ..config import _geometry_from_mapping
from ..data.dataset import _pose, _rigid_transform_class
from ..data.geometry import load_nifti_volume
from ..rendering.drr import _prepare_legacy_imports


def fft_case(case):
    settings = case.training_params.get("fft", {})
    geometry = _geometry_from_mapping({**vars(case.geometry), **settings.get("geometry", {})})
    params = {**case.training_params, **settings}
    return replace(
        case,
        geometry=geometry,
        training_params=params,
        train_model=settings.get("model", case.train_model or case.model),
    )


def build_fft_dataset(case):
    """Compute centred poses before resampling, as in the DICOM trainers."""
    dataset = build_dataset(case)
    unscaled = replace(case.geometry, factors=(1.0, 1.0, 1.0), post_flip_axes=())
    volume, spacing, _affine = load_nifti_volume(case.ct_path, unscaled)
    shape = np.asarray(volume.shape, dtype=np.float64)
    offsets = np.asarray([unscaled.x_offset, unscaled.y_offset, unscaled.z_offset])
    center = (shape - offsets) * spacing / 2.0
    rigid = _rigid_transform_class()
    dataset.center_pose = _pose(rigid, [0.0, 0.0, 0.0], center)
    dataset.back_pose = _pose(rigid, [0.0, 0.0, 0.0], -center)
    settings = case.training_params.get("fft", {})
    pose_file = settings.get("isocenter_pose_file")
    if pose_file:
        values = json.loads((case.gt_pose_path / pose_file).read_text(encoding="utf-8"))["pose_params"]
        dataset.isocenter_pose = rigid(
            torch.tensor([values[:3]], dtype=torch.float32),
            torch.tensor([values[3:]], dtype=torch.float32),
            parameterization="so3_log_map",
        )
    else:
        dataset.isocenter_pose = _pose(rigid, [-np.pi / 2, 0.0, np.pi / 2], center)
    return dataset


def random_fft_offset(batch_size, device):
    _prepare_legacy_imports()
    from ours.case.my_util2 import get_random_offset

    return get_random_offset(batch_size, device)


def fft_transforms(height):
    _prepare_legacy_imports()
    from ours.utils.CT_dataset import Transforms

    return Transforms(height, radius=119)


def prepare_fft_batch(image, bone, transforms, cutoff_ratio, augmenter=None, tube_probability=0.7):
    _prepare_legacy_imports()
    from ours.fft import high_pass_gaussian

    bone = transforms(bone.detach().float(), reverse=False)
    bone = (bone - bone.min()) / (bone.max() - bone.min())
    target_mask = torch.tanh(50 * bone)
    target_image = transforms(image).float()
    model_input = transforms(high_pass_gaussian(target_image, cutoff_ratio), reverse=False).float()
    tube_applied = augmenter is not None and random.random() < tube_probability
    if tube_applied:
        model_input, tube = augmenter.augment(model_input)
        target_mask = target_mask * tube
    model_input = transforms(model_input, reverse=False).float()
    return target_image, model_input, target_mask, tube_applied


def fft_scheduler(optimizer, epochs, batches_per_epoch, warmup_steps):
    from pytorch_transformers.optimization import WarmupCosineSchedule

    total = epochs * batches_per_epoch - warmup_steps
    if total <= warmup_steps:
        raise ValueError("FFT schedule requires epochs * batches-per-epoch > 2 * warmup-steps")
    return WarmupCosineSchedule(optimizer, warmup_steps, total)
