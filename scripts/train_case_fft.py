#!/usr/bin/env python3
"""Train one configured case with the legacy FFT/tube-augmentation recipe.

This is intentionally separate from ``train_case.py``: the regular trainer
uses the current image preprocessing, whereas this entrypoint preserves the
ysy/sxh high-pass DRR training path from ``diffpose/ours``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFPOSE_ROOT = PROJECT_ROOT.parent / "diffpose"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
for legacy_path in (DIFFPOSE_ROOT, DIFFPOSE_ROOT / "ours"):
    if str(legacy_path) not in sys.path:
        sys.path.insert(0, str(legacy_path))

from ours.utils.mask_data import MaskDataAugmentation  # noqa: E402
from vprn.cli.common import build_model, case_snapshot, checkpoint_run_dir  # noqa: E402
from vprn.config import load_case  # noqa: E402
from vprn.evaluation.artifacts import _json_safe  # noqa: E402
from vprn.registration.initialization import forward_model  # noqa: E402
from vprn.registration.optimizers import create_scheduler  # noqa: E402
from vprn.rendering.drr import create_drr  # noqa: E402
from vprn.training.fft_recipe import (  # noqa: E402
    build_fft_dataset, fft_case, fft_scheduler, fft_transforms,
    prepare_fft_batch, random_fft_offset,
)
from vprn.training.checkpoint import save_checkpoint  # noqa: E402
from vprn.training.losses import registration_loss  # noqa: E402
from vprn.training.trainer import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one configured clinical case with legacy FFT preprocessing."
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "cases.json")
    parser.add_argument("--run-id", default="train-fft")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1000, help="Final epoch label; fresh runs execute epochs 0 through this value.")
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--drr-patch-size",
        type=int,
        help="DRR raycasting patch edge length; defaults to height // 2, matching legacy FFT trainers.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=50,
        help="Save an epoch checkpoint every N epochs; last.ckpt is saved every epoch.",
    )
    parser.add_argument("--accumulate-step", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--disable-agc", action="store_true", help="Disable the original adaptive gradient clipping.")
    parser.add_argument(
        "--scheduler", choices=["none", "step", "cosine", "warmup_cosine"], default="warmup_cosine"
    )
    parser.add_argument("--scheduler-step-size", type=int, default=25)
    parser.add_argument("--scheduler-gamma", type=float, default=0.9)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Defaults to five epochs (5 * batches-per-epoch), matching the legacy FFT trainers.",
    )
    parser.add_argument("--cutoff-min", type=float, default=0.01)
    parser.add_argument("--cutoff-max", type=float, default=0.33)
    parser.add_argument("--contrast-min", type=float, default=0.5)
    parser.add_argument("--contrast-max", type=float, default=8.0)
    parser.add_argument("--tube-dir", type=Path, default=DIFFPOSE_ROOT / "ours" / "tube")
    parser.add_argument("--tube-augmentation-prob", type=float, default=0.7)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.height <= 0 or args.batch_size <= 0 or args.batches_per_epoch <= 0 or args.epochs < 0:
        raise ValueError("height, batch-size, and batches-per-epoch must be positive; epochs must be non-negative")
    if not 0.0 <= args.tube_augmentation_prob <= 1.0:
        raise ValueError("tube-augmentation-prob must be in [0, 1]")
    if not 0.0 < args.cutoff_min <= args.cutoff_max:
        raise ValueError("cutoff bounds must satisfy 0 < cutoff-min <= cutoff-max")
    if not 0.0 < args.contrast_min <= args.contrast_max:
        raise ValueError("contrast bounds must satisfy 0 < contrast-min <= contrast-max")
    if args.save_every_epochs <= 0:
        raise ValueError("save-every-epochs must be positive")
    if args.accumulate_step <= 0 or args.batches_per_epoch % args.accumulate_step:
        raise ValueError("batches-per-epoch must be divisible by accumulate-step")


def main() -> int:
    args = parse_args()
    _validate_args(args)
    drr_patch_size = args.drr_patch_size or args.height // 2
    if drr_patch_size <= 0 or args.height % drr_patch_size != 0:
        raise ValueError("drr-patch-size must be a positive divisor of height")
    case = fft_case(load_case(args.case, args.config))
    dataset = build_fft_dataset(case)
    checkpoint_dir = checkpoint_run_dir(PROJECT_ROOT / "checkpoints", case.case_id, args.run_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_model = case.train_model or case.model
    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = 5 * args.batches_per_epoch if args.scheduler == "warmup_cosine" else 0
    config_snapshot = case_snapshot(
        case,
        role="train",
        device=args.device,
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        scheduler=args.scheduler,
        warmup_steps=warmup_steps,
        learning_rate=args.learning_rate,
        accumulate_step=args.accumulate_step,
        adaptive_gradient_clipping=not args.disable_agc,
        gradient_clip_norm=args.gradient_clip_norm,
        epoch_numbering="zero_based_inclusive",
        drr_patch_size=drr_patch_size,
        fft={
            "cutoff_min": args.cutoff_min,
            "cutoff_max": args.cutoff_max,
            "contrast_min": args.contrast_min,
            "contrast_max": args.contrast_max,
            "tube_dir": str(args.tube_dir),
            "tube_augmentation_prob": args.tube_augmentation_prob,
        },
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(_json_safe(config_snapshot), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(
            f"[dry-run] {case.case_id}: CT={dataset.nii_path} X-ray frames={len(dataset)} "
            f"FFT cutoff=[{args.cutoff_min}, {args.cutoff_max}]"
        )
        return 0

    augmenter = None
    if args.tube_augmentation_prob > 0.0:
        if not args.tube_dir.is_dir():
            raise FileNotFoundError(f"tube mask directory does not exist: {args.tube_dir}")
        augmenter = MaskDataAugmentation(args.tube_dir, resize_to=(args.height, args.height))

    model = build_model(case, args.device, role="train")
    bone_threshold = float(case.training_params.get("bone_threshold", 300.0))
    drr = create_drr(
        dataset,
        height=args.height,
        device=args.device,
        bone_threshold=bone_threshold,
        patch_size=drr_patch_size,
        delx=dataset.delx * 512 / args.height,
    )
    drr_bone = create_drr(
        dataset,
        height=args.height,
        device=args.device,
        bone=True,
        bone_threshold=bone_threshold,
        patch_size=drr_patch_size,
        delx=dataset.delx * 512 / args.height,
    )
    transforms = fft_transforms(args.height)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    resume_payload = None
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location=args.device, weights_only=False)
        # Resuming another architecture must fail before any training update.
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        if isinstance(resume_payload, dict) and "optimizer_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
    start_epoch = int(resume_payload["epoch"]) + 1 if resume_payload is not None else 0
    if start_epoch > args.epochs:
        raise ValueError("resume checkpoint already reached --epochs; choose a higher final epoch")

    from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
    from diffpose.metrics import DoubleGeodesic, GeodesicSE3

    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double_geodesic = DoubleGeodesic(dataset.sdr)
    contrast_distribution = torch.distributions.Uniform(args.contrast_min, args.contrast_max)
    cutoff_distribution = torch.distributions.Uniform(args.cutoff_min, args.cutoff_max)

    def step_fn(current_model, _batch):
        contrast = contrast_distribution.sample().item()
        cutoff_ratio = cutoff_distribution.sample().item()
        offset = random_fft_offset(args.batch_size, args.device)
        isocenter = dataset.isocenter_pose.to(args.device)
        pose = (
            isocenter.compose(dataset.back_pose.to(args.device))
            .compose(offset)
            .compose(dataset.center_pose.to(args.device))
        )

        image = drr(None, None, None, pose=pose, bone_attenuation_multiplier=contrast)
        bone = drr_bone(None, None, None, pose=pose)
        image, image_input, target_mask, _tube_applied = prepare_fft_batch(
            image, bone, transforms, cutoff_ratio, augmenter, args.tube_augmentation_prob,
        )

        predicted_offset, predicted_mask = forward_model(current_model, image_input, target_mask)
        predicted_pose = isocenter.compose(predicted_offset)
        predicted_image = transforms(drr(None, None, None, pose=predicted_pose).float())
        return registration_loss(
            predicted_pose,
            pose,
            predicted_image,
            image,
            metric=metric,
            geodesic=geodesic,
            double_geodesic=double_geodesic,
            predicted_mask=predicted_mask,
            target_mask=target_mask,
            mask_weight=float(case.training_params.get("mask_weight", 10.0)),
        )

    if args.scheduler == "warmup_cosine":
        scheduler = fft_scheduler(optimizer, args.epochs, args.batches_per_epoch, warmup_steps)
    else:
        scheduler = create_scheduler(
            optimizer, kind=args.scheduler, step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma, total_steps=args.epochs * args.batches_per_epoch,
        )
    if (
        scheduler is not None
        and isinstance(resume_payload, dict)
        and "scheduler_state_dict" in resume_payload
    ):
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        # Scheduler construction resets lr to step zero. Restore the saved
        # optimizer rates so the first resumed update uses the correct rate.
        for group, saved in zip(optimizer.param_groups, resume_payload["optimizer_state_dict"]["param_groups"]):
            group["lr"] = saved["lr"]
    from timm.utils.agc import adaptive_clip_grad

    trainer = Trainer(
        model,
        optimizer,
        step_fn,
        scheduler=scheduler,
        device=args.device,
        accumulate_steps=args.accumulate_step,
        gradient_clip_norm=args.gradient_clip_norm,
        gradient_clip_fn=None if args.disable_agc else adaptive_clip_grad,
        skip_nan_loss=True,
        flush_accumulation=False,
    )
    best_loss = float("inf")
    best_checkpoint = checkpoint_dir / "best.ckpt"
    if args.resume is not None and best_checkpoint.is_file():
        best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(best_payload, dict) and "loss" in best_payload:
            best_loss = float(best_payload["loss"])
    elif resume_payload is not None:
        best_loss = float(resume_payload.get("loss", float("inf")))

    history_path = checkpoint_dir / "train_history.json"
    previous_history = json.loads(history_path.read_text()) if args.resume and history_path.is_file() else []

    checkpoint_metadata = {
        "case": case.case_id,
        "model_class": train_model,
        **case.model_params,
        "epoch_numbering": "zero_based_inclusive",
        "height": args.height,
        "batch_size": args.batch_size,
        "batches_per_epoch": args.batches_per_epoch,
        "n_batches_per_epoch": args.batches_per_epoch,
        "n_epochs": args.epochs,
        "drr_patch_size": drr_patch_size,
    }

    def write_history(history: list[dict[str, float | int]]) -> None:
        (checkpoint_dir / "train_history.json").write_text(
            json.dumps(_json_safe(previous_history + history), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save_state(path: Path, completed_epoch: int, loss: float) -> None:
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler=scheduler,
            epoch=completed_epoch,
            loss=loss,
            **checkpoint_metadata,
        )

    def on_epoch_end(
        relative_epoch: int,
        record: dict[str, float | int],
        partial_history: list[dict[str, float | int]],
    ) -> None:
        nonlocal best_loss
        completed_epoch = start_epoch + relative_epoch
        record["epoch"] = completed_epoch
        loss = float(record["loss"])
        if loss < best_loss:
            best_loss = loss
            save_state(best_checkpoint, completed_epoch, loss)
        save_state(checkpoint_dir / "last.ckpt", completed_epoch, loss)
        write_history(partial_history)
        if completed_epoch % args.save_every_epochs == 0:
            save_state(checkpoint_dir / f"epoch{completed_epoch:04d}.ckpt", completed_epoch, loss)
            print(f"Saved checkpoint at epoch {completed_epoch}: loss={loss:.6f}")

    history = trainer.fit(
        [None] * args.batches_per_epoch,
        epochs=args.epochs - start_epoch + 1,
        progress=True,
        on_epoch_end=on_epoch_end,
    )
    write_history(history)
    final_loss = float(history[-1]["loss"]) if history else float("nan")
    final_epoch = args.epochs
    save_state(checkpoint_dir / "last.ckpt", final_epoch, final_loss)
    if not best_checkpoint.is_file():
        save_state(best_checkpoint, final_epoch, final_loss)
    print(json.dumps(history[-1] if history else {"epochs": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
