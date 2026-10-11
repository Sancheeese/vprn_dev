#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vprn.cli.common import build_dataset, build_dataset_and_model, case_snapshot, model_prediction_record  # noqa: E402
from vprn.config import load_case  # noqa: E402
from vprn.evaluation.artifacts import ArtifactStore  # noqa: E402
from vprn.models.registry import load_checkpoint  # noqa: E402
from vprn.rendering.transforms import ImageTransforms  # noqa: E402
from vprn.registration.engine import RegistrationEngine  # noqa: E402
from vprn.registration.backends import refine_pose  # noqa: E402
from vprn.rendering.drr import create_drr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register one configured clinical case.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "cases.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-id", default="register")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--method", choices=["pose_regression", "sparse", "gradient", "cmaes"])
    parser.add_argument("--iterations", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = load_case(args.case, args.config)
    if args.dry_run:
        dataset = build_dataset(case)
        print(f"[dry-run] {case.case_id}: CT={dataset.nii_path} X-ray frames={len(dataset)}")
        return 0
    dataset, model = build_dataset_and_model(case, args.device, role="register")
    checkpoint = args.checkpoint or case.register_checkpoint or case.checkpoint
    metadata = load_checkpoint(model, checkpoint, device=args.device)
    store = ArtifactStore(PROJECT_ROOT / "runs", case.case_id, args.run_id, include_checkpoints=False)
    store.ensure()
    method = args.method or case.registration_method
    iterations = args.iterations
    if iterations is None:
        iterations = int(case.registration_params.get("iterations", 50))
    registration_params = case.registration_params
    store.save_config(case_snapshot(
        case,
        role="register",
        checkpoint=str(checkpoint),
        checkpoint_metadata=metadata,
        method=method,
        iterations=iterations,
        device=args.device,
    ))
    transforms = ImageTransforms(256)
    engine = RegistrationEngine(
        model,
        dataset,
        device=args.device,
        preprocess=lambda image: transforms(image, reverse=True),
        mask_preprocess=lambda mask: transforms(mask, reverse=False),
    )
    drr = drr_bone = None
    if method in {"sparse", "gradient", "cmaes"}:
        drr = create_drr(dataset, height=256, device=args.device)
        if method in {"sparse", "gradient"}:
            drr_bone = create_drr(dataset, height=256, device=args.device, bone=True)
    stop = len(dataset) if args.stop is None else min(args.stop, len(dataset))
    rows = []
    for index in range(args.start, stop):
        result = engine.run(index)
        prediction = result.prediction
        if method in {"sparse", "gradient", "cmaes"}:
            target_image = transforms(dataset.xray.read_pixels(index), reverse=True).to(args.device)
            refine_kwargs = {
                "iterations": iterations,
            }
            if method in {"sparse", "gradient"}:
                for key in ("rotation_lr", "translation_lr", "optimizer", "maximize"):
                    if key in registration_params:
                        refine_kwargs[key] = registration_params[key]
            elif method == "cmaes":
                for key in ("sigma0", "population_size"):
                    if key in registration_params:
                        refine_kwargs[key] = registration_params[key]
            prediction = refine_pose(
                prediction,
                target_image,
                drr,
                drr_bone,
                method=method,
                parameterization=case.model_params.get("parameterization", "se3_log_map"),
                convention=case.model_params.get("convention"),
                **refine_kwargs,
            )
        row = model_prediction_record(index, prediction)
        try:
            row["tre"] = float(dataset.calc_tre(index, prediction).detach().cpu())
        except (FileNotFoundError, ValueError, RuntimeError):
            pass
        rows.append(row)
    csv_path = store.write_csv("registration/poses.csv", rows)
    store.write_json("registration/poses.json", rows)
    print(json.dumps({"case": case.case_id, "rows": len(rows), "poses": str(csv_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
