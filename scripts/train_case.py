#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vprn.cli.common import build_dataset, build_model, case_snapshot, checkpoint_run_dir  # noqa: E402
from vprn.config import load_case  # noqa: E402
from vprn.evaluation.artifacts import _json_safe  # noqa: E402
from vprn.rendering.drr import create_drr  # noqa: E402
from vprn.rendering.transforms import ImageTransforms  # noqa: E402
from vprn.registration.initialization import forward_model  # noqa: E402
from vprn.training.checkpoint import save_checkpoint  # noqa: E402
from vprn.training.losses import registration_loss  # noqa: E402
from vprn.training.trainer import Trainer  # noqa: E402
from vprn.registration.optimizers import create_scheduler  # noqa: E402
from vprn.models.registry import load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one configured clinical case.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "cases.json")
    parser.add_argument("--run-id", default="train")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--drr-patch-size",
        type=int,
        help="DRR raycasting patch edge length; defaults to height // 2, matching legacy trainers.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--accumulate-step", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine", "warmup_cosine"], default="none")
    parser.add_argument("--scheduler-step-size", type=int, default=25)
    parser.add_argument("--scheduler-gamma", type=float, default=0.9)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    drr_patch_size = args.drr_patch_size or args.height // 2
    if drr_patch_size <= 0 or args.height % drr_patch_size != 0:
        raise ValueError("drr-patch-size must be a positive divisor of height")
    case = load_case(args.case, args.config)
    dataset = build_dataset(case)
    checkpoint_dir = checkpoint_run_dir(PROJECT_ROOT / "checkpoints", case.case_id, args.run_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_model = case.train_model or case.model
    config_snapshot = case_snapshot(
        case,
        role="train",
        device=args.device,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        scheduler=args.scheduler,
        drr_patch_size=drr_patch_size,
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(_json_safe(config_snapshot), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(f"[dry-run] {case.case_id}: CT={dataset.nii_path} X-ray frames={len(dataset)}")
        return 0

    model = build_model(case, args.device, role="train")

    bone_threshold = float(case.training_params.get("bone_threshold", 300.0))
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location=args.device, weights_only=False)
        load_checkpoint(model, args.resume, device=args.device)
        if isinstance(resume_payload, dict) and "optimizer_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
    from diffpose.metrics import DoubleGeodesic, GeodesicSE3

    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double_geodesic = DoubleGeodesic(dataset.sdr)

    def step_fn(current_model, _batch):
        from vprn.cli.legacy import random_offset

        offset = random_offset(args.batch_size, args.device)
        isocenter = dataset.isocenter_pose.to(args.device)
        pose = isocenter.compose(dataset.back_pose.to(args.device)).compose(offset).compose(dataset.center_pose.to(args.device))
        image = transforms(drr(None, None, None, pose=pose), reverse=True)
        bone = transforms(drr_bone(None, None, None, pose=pose), reverse=False)
        bone = (bone - bone.amin()) / (bone.amax() - bone.amin() + 1e-6)
        bone = torch.tanh(50.0 * bone)
        predicted_offset, predicted_mask = forward_model(current_model, image, bone)
        predicted_pose = isocenter.compose(predicted_offset)
        predicted_image = transforms(drr(None, None, None, pose=predicted_pose), reverse=True)
        return registration_loss(
            predicted_pose,
            pose,
            predicted_image,
            image,
            metric=metric,
            geodesic=geodesic,
            double_geodesic=double_geodesic,
            predicted_mask=predicted_mask,
            target_mask=bone,
            mask_weight=float(case.training_params.get("mask_weight", 10.0)),
        )

    total_steps = args.epochs * args.batches_per_epoch
    scheduler = create_scheduler(
        optimizer,
        kind=args.scheduler,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
        total_steps=total_steps if args.scheduler != "none" else None,
        warmup_steps=args.warmup_steps,
    )
    trainer = Trainer(
        model,
        optimizer,
        step_fn,
        scheduler=scheduler,
        device=args.device,
        accumulate_steps=args.accumulate_step,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    history = trainer.fit([None] * args.batches_per_epoch, epochs=args.epochs, progress=True)
    (checkpoint_dir / "train_history.json").write_text(
        json.dumps(_json_safe(history), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_checkpoint(
        checkpoint_dir / "last.ckpt",
        model,
        optimizer,
        scheduler=scheduler,
        epoch=int((resume_payload or {}).get("epoch", 0)) + args.epochs,
        loss=history[-1]["loss"] if history else float("nan"),
        case=case.case_id,
        model_name=train_model,
    )
    save_checkpoint(
        checkpoint_dir / "best.ckpt",
        model,
        optimizer,
        scheduler=scheduler,
        epoch=int((resume_payload or {}).get("epoch", 0)) + args.epochs,
        loss=history[-1]["loss"] if history else float("nan"),
        case=case.case_id,
        model_name=train_model,
    )
    print(json.dumps(history[-1] if history else {"epochs": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
