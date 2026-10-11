#!/usr/bin/env python3
"""Save visual previews of one normal or FFT-training batch without updates."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFPOSE_ROOT = PROJECT_ROOT.parent / "diffpose"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
for legacy_path in (DIFFPOSE_ROOT, DIFFPOSE_ROOT / "ours"):
    if str(legacy_path) not in sys.path:
        sys.path.insert(0, str(legacy_path))

from ours.fft import high_pass_gaussian  # noqa: E402
from ours.utils.mask_data import MaskDataAugmentation  # noqa: E402
from vprn.cli.common import build_dataset, build_model  # noqa: E402
from vprn.cli.legacy import random_offset  # noqa: E402
from vprn.config import load_case  # noqa: E402
from vprn.models.registry import load_checkpoint  # noqa: E402
from vprn.registration.initialization import forward_model  # noqa: E402
from vprn.rendering.drr import create_drr  # noqa: E402
from vprn.rendering.transforms import ImageTransforms  # noqa: E402
from vprn.training.losses import registration_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the exact normal or FFT input pipeline used by the case trainers."
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "cases.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["normal", "fft"], default="fft")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--drr-patch-size",
        type=int,
        help="DRR raycasting patch edge length; defaults to height // 2, matching legacy trainers.",
    )
    parser.add_argument("--cutoff-min", type=float, default=0.01)
    parser.add_argument("--cutoff-max", type=float, default=0.33)
    parser.add_argument("--contrast-min", type=float, default=0.5)
    parser.add_argument("--contrast-max", type=float, default=8.0)
    parser.add_argument("--tube-dir", type=Path, default=DIFFPOSE_ROOT / "ours" / "tube")
    parser.add_argument("--tube-augmentation-prob", type=float, default=0.7)
    parser.add_argument("--checkpoint", type=Path, help="Optional trained checkpoint for meaningful prediction panels.")
    parser.add_argument("--num-previews", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional reproducibility seed. Omit it to sample a new training-like preview each run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to previews/<case>/<mode>; existing preview PNGs in that directory are replaced.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.height <= 0 or args.num_previews <= 0:
        raise ValueError("batch-size, height, and num-previews must be positive")
    if not 0.0 <= args.tube_augmentation_prob <= 1.0:
        raise ValueError("tube-augmentation-prob must be in [0, 1]")
    if not 0.0 < args.cutoff_min <= args.cutoff_max:
        raise ValueError("cutoff bounds must satisfy 0 < cutoff-min <= cutoff-max")
    if not 0.0 < args.contrast_min <= args.contrast_max:
        raise ValueError("contrast bounds must satisfy 0 < contrast-min <= contrast-max")


def _as_gray_image(image: torch.Tensor, index: int) -> Image.Image:
    array = image[index].detach().float().cpu().squeeze().numpy()
    array = (array - array.min()) / (array.max() - array.min() + 1e-6)
    return Image.fromarray(np.round(array * 255).astype(np.uint8), mode="L").convert("RGB")


def _save_montage(path: Path, panels: list[tuple[str, torch.Tensor]], index: int) -> None:
    panel_images = [_as_gray_image(image, index) for _label, image in panels]
    width, height = panel_images[0].size
    label_height = 28
    montage = Image.new("RGB", (width * len(panel_images), height + label_height), "white")
    draw = ImageDraw.Draw(montage)
    for panel_index, ((label, _image), panel) in enumerate(zip(panels, panel_images)):
        x = panel_index * width
        montage.paste(panel, (x, label_height))
        draw.text((x + 6, 7), label, fill="black")
    montage.save(path)


def main() -> int:
    args = parse_args()
    _validate_args(args)
    if args.mode == "fft" and not args.tube_dir.is_dir() and args.tube_augmentation_prob > 0.0:
        raise FileNotFoundError(f"tube mask directory does not exist: {args.tube_dir}")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    case = load_case(args.case, args.config)
    output_dir = args.output_dir or PROJECT_ROOT / "previews" / case.case_id / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_preview in output_dir.glob("sample_*.png"):
        stale_preview.unlink()
    dataset = build_dataset(case)
    model = build_model(case, args.device, role="train")
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, device=args.device)
    model.train()
    bone_threshold = float(case.training_params.get("bone_threshold", 300.0))
    drr_patch_size = args.drr_patch_size or args.height // 2
    if drr_patch_size <= 0 or args.height % drr_patch_size != 0:
        raise ValueError("drr-patch-size must be a positive divisor of height")
    drr = create_drr(
        dataset,
        height=args.height,
        device=args.device,
        bone_threshold=bone_threshold,
        patch_size=drr_patch_size,
    )
    drr_bone = create_drr(
        dataset,
        height=args.height,
        device=args.device,
        bone=True,
        bone_threshold=bone_threshold,
        patch_size=drr_patch_size,
    )
    transforms = ImageTransforms(args.height)
    augmenter = (
        MaskDataAugmentation(args.tube_dir, resize_to=(args.height, args.height))
        if args.mode == "fft" and args.tube_augmentation_prob > 0.0
        else None
    )

    from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
    from diffpose.metrics import DoubleGeodesic, GeodesicSE3

    contrast = None
    cutoff_ratio = None
    isocenter = dataset.isocenter_pose.to(args.device)
    pose = (
        isocenter.compose(dataset.back_pose.to(args.device))
        .compose(random_offset(args.batch_size, args.device))
        .compose(dataset.center_pose.to(args.device))
    )
    with torch.no_grad():
        if args.mode == "fft":
            contrast = torch.distributions.Uniform(args.contrast_min, args.contrast_max).sample().item()
            cutoff_ratio = torch.distributions.Uniform(args.cutoff_min, args.cutoff_max).sample().item()
            target_image = transforms(
                drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast), reverse=False
            ).to(torch.float32)
        else:
            target_image = transforms(drr(None, None, None, pose=pose), reverse=True).to(torch.float32)
        bone = transforms(drr_bone(None, None, None, pose=pose), reverse=False)
        bone = (bone - bone.amin()) / (bone.amax() - bone.amin() + 1e-6)
        bone = torch.tanh(50.0 * bone)
        tube_applied = False
        if args.mode == "fft":
            model_input = transforms(high_pass_gaussian(target_image, cutoff_ratio), reverse=False).to(torch.float32)
            target_mask = bone
            tube_applied = augmenter is not None and random.random() < args.tube_augmentation_prob
            if tube_applied:
                model_input, tube = augmenter.augment(model_input)
                target_mask = target_mask * tube
            model_input = transforms(model_input, reverse=False).to(torch.float32)
            predicted_image_reverse = False
        else:
            model_input = target_image
            target_mask = bone
            predicted_image_reverse = True
        predicted_offset, predicted_mask = forward_model(model, model_input, target_mask)
        if predicted_mask is None:
            raise RuntimeError("configured model did not return a mask prediction")
        predicted_pose = isocenter.compose(predicted_offset)
        predicted_image = transforms(
            drr(None, None, None, pose=predicted_pose), reverse=predicted_image_reverse
        )
        loss = registration_loss(
            predicted_pose,
            pose,
            predicted_image,
            target_image,
            metric=MultiscaleNormalizedCrossCorrelation2d(eps=1e-4),
            geodesic=GeodesicSE3(),
            double_geodesic=DoubleGeodesic(dataset.sdr),
            predicted_mask=predicted_mask,
            target_mask=target_mask,
            mask_weight=float(case.training_params.get("mask_weight", 10.0)),
        )

    panels = [
        ("target DRR", target_image),
        ("FFT input" if args.mode == "fft" else "normal input", model_input),
        ("bone/tube target", target_mask),
        ("predicted DRR", predicted_image),
        ("predicted mask", predicted_mask),
    ]
    for index in range(min(args.num_previews, args.batch_size)):
        _save_montage(output_dir / f"sample_{index:02d}.png", panels, index)
    metadata = {
        "case": case.case_id,
        "mode": args.mode,
        "device": args.device,
        "batch_size": args.batch_size,
        "height": args.height,
        "drr_patch_size": drr_patch_size,
        "bone_threshold": bone_threshold,
        "seed": args.seed,
        "contrast": contrast,
        "cutoff_ratio": cutoff_ratio,
        "tube_applied": tube_applied,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "loss_before_training": float(loss.cpu()),
        "panels": [label for label, _image in panels],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))
    print(f"Saved {min(args.num_previews, args.batch_size)} preview image(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
