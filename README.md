# VPRN Clinical Registration Pipeline

This directory is an independent, configuration-driven version of the active
clinical CT-X-ray registration workflows. Nothing under the original
`diffpose/` tree is modified by this project.

## Cases

The initial data inventory contains only `ysy`, `sxh`, `wfl`, `xyl`, and `zyl`.
CT volumes are NIfTI files. X-ray frames and segmentation files retain their
original formats. Per-case CT series, axis transforms, cropping, scaling,
isocenter offsets, pose source, model, and checkpoint are defined in
`configs/cases.json`.

## Layout

```
vprn/
├── configs/cases.json       # one manifest: inputs, geometry, pose, models
├── data/<case>/             # selected CT NIfTI, X-ray DICOM, masks, GT poses
├── checkpoints/<case>/<run-id>/ # trained model checkpoints and training history
├── src/vprn/                 # data, rendering, models, training, registration, evaluation
├── scripts/                 # convert_ct, train_case, register_case, evaluate_case
└── runs/<case>/<run-id>/    # registration outputs and evaluation metrics
```

## Commands

Run commands from this directory with the `mr2ct` conda environment:

```bash
PYTHONPATH=src conda run -n mr2ct python scripts/convert_ct.py \
  --case xyl \
  --config configs/cases.json \
  --source-dir ../diffpose/ours/data/liwei/许玉露/CT/XuYuLu/20240326172117.697000/3

PYTHONPATH=src conda run -n mr2ct python scripts/train_case.py \
  --case sxh --run-id train-001

PYTHONPATH=src conda run -n mr2ct python scripts/register_case.py \
  --case sxh --run-id register-001

PYTHONPATH=src conda run -n mr2ct python scripts/evaluate_case.py \
  --case sxh --run-id register-001
```

Use `--dry-run` on the training command to validate paths and model setup
without rendering or updating a checkpoint. GPU execution is required for the
full DRR training/register workloads.

The registration command accepts `--method pose_regression`, `--method sparse`,
`--method gradient`, or `--method cmaes`. `sparse` and `gradient` use the
legacy differentiable sparse renderer with separate rotation/translation
learning rates; `cmaes` uses the same DRR objective through the optional `cma`
dependency. The default method comes from each case configuration.

Training options such as `--accumulate-step`, `--gradient-clip-norm`,
`--scheduler`, and `--resume` are shared across cases. The configured
`train_model` and `register_model` fields are independent, which is required
for WFL where training used `PoseRegressorGSpDeco` and registration used
`PoseRegressorCoeSpDeco`.

## FFT Training

`scripts/train_case_fft.py` follows the original ysy/sxh AddSpDeco FFT recipe:
the legacy random pose sampler, inverted DRRs, circle radius 119, Gaussian
high-pass cutoff 0.01-0.33, contrast 0.5-8.0, tube augmentation probability 0.7,
NCC/pose/mask loss, adaptive gradient clipping, and the original
`pytorch_transformers` warmup/cosine scheduler.

Case-specific FFT overrides live in `training_params.fft` in `configs/cases.json`.
YSY uses CT factors `[0.7, 1.25, 0.7]`; SXH uses `[0.6, 0.6, 1.0]` and its
`gt_pose/sxh.json` isocentre. Centre/back poses are computed before resampling.
ZYL FFT training selects AddSpDeco and retains its own CT geometry and 350 HU
bone threshold. Its converted NIfTI already reverses the DICOM slice order;
the FFT override disables the extra axis-0 flip to match the original ZYL
DICOM volume. Ordinary training and registration use the existing configuration.

```bash
PYTHONPATH=src conda run --no-capture-output -n mr2ct python scripts/train_case_fft.py \
  --case zyl --run-id train-addspdeco-fft --device cuda:0

PYTHONPATH=src conda run --no-capture-output -n mr2ct python scripts/preview_fft_training.py \
  --case zyl --mode fft --device cuda:0
```

Defaults match the original entrypoints: batch size 8, 100 batches per epoch,
Adam learning rate 0.001, and accumulation 1. `--epochs 1000` denotes the final
epoch label and executes epochs 0-1000 on a fresh run. `last.ckpt` is saved every
epoch and numbered snapshots every 50 epochs. For a short check, use
`--epochs 0 --batches-per-epoch 1 --batch-size 1 --scheduler none`.

`--resume checkpoints/zyl/train-addspdeco-fft/last.ckpt` resumes at the next
epoch through the specified final epoch. This avoids repeating the saved epoch
and the original scripts' automatic retraining/retry loops. Resume requires
matching model weights. The existing ZYL CoeSpDeco checkpoint is not an
AddSpDeco initialization. FFT-trained models also require matching FFT
preprocessing for registration; the current registration command does not yet
apply that preprocessing.

## Results

Training artifacts are isolated under `checkpoints/<case>/<run-id>/` with a
config snapshot, `best.ckpt`, `last.ckpt`, and `train_history.json`.
Registration artifacts are isolated under `runs/<case>/<run-id>/` with a
config snapshot, registration poses, metrics, figures, and logs. This keeps
model weights separate from inference results.

Registration CSV rows contain the six pose components (`rx`, `ry`, `rz`, `tx`,
`ty`, `tz`) and include `tre` when the case has fiducials. `evaluate_case.py`
filters non-numeric metadata and writes the mean summary to
`metrics/summary.json`.

## Verification

```bash
PYTHONPATH=src conda run -n mr2ct pytest -q tests
```

The geometry parity test checks all five configured CT pipelines against the
legacy volume shapes and spacings. Additional parity tests should be added
before changing axis transforms or pose parameterization.
