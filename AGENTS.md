# VPRN Working Rules

## Scope and Orientation

- This file applies to all work under `/home/zsr/project/vprn`.
- Before changing VPRN code, read `README.md`, `configs/cases.json`, and the
  parent project guide at `/home/zsr/project/PROJECT_DIRECTORY_GUIDE.md`.
- Before changing CT geometry, pose handling, fiducials, or registration output
  formats, also read `/home/zsr/project/POSE_CONVENTIONS.md`.
- Use the `mr2ct` conda environment for Python commands.
- Do not modify `/home/zsr/project/diffpose/` or `/home/zsr/project/mrct/`
  while working on VPRN. VPRN may read legacy implementations from them, but
  changes belong in VPRN unless the user explicitly expands the scope.

## Project Purpose

VPRN is the configuration-driven clinical CT-to-X-ray registration pipeline
for exactly these active cases: `ysy`, `sxh`, `wfl`, `xyl`, and `zyl`.

Case-specific CT selection, geometry, poses, models, checkpoints, and
registration parameters belong in `configs/cases.json`. Do not create a new
case-specific training, dataset, or registration script when a configuration
entry can represent the difference.

## Directory Contract

```text
configs/                         Versioned case configuration.
data/<case>/                     Local clinical input data; Git-ignored.
  ct/ct.nii                      Selected CT converted to NIfTI.
  xray/                          Original X-ray DICOM frames.
  seg/                           Original segmentation files.
  gt_pose/                       Ground-truth/manual pose files.
  fid_<case>.json                Fiducials when available.

checkpoints/<case>/<run-id>/     Local training artifacts; Git-ignored.
  config.json                    Training configuration snapshot.
  best.ckpt                      Best model for that training run.
  last.ckpt                      Most recent model for that training run.
  train_history.json             Per-epoch training loss history.

runs/<case>/<run-id>/            Registration/evaluation artifacts; versioned when selected.
  config.json                    Registration configuration snapshot.
  registration/poses.csv         Final per-frame registered poses.
  registration/poses.json        JSON copy of the registered poses.
  metrics/summary.json           Evaluation summary for this registration run.
  figures/                       Optional registration visualizations.
  logs/                          Optional registration logs.

scripts/                         User-facing CT conversion, training, registration,
                                 and evaluation commands.
src/vprn/                        Reusable pipeline code.
tests/                           Automated tests; test artifacts must use pytest tmp paths.
```

`runs/` is only for real registration or evaluation output. Do not place unit
test, smoke-test, temporary checkpoint, or training output there. Clear an
accidental test run from `runs/` rather than treating it as a result.

## Run IDs and Default Artifacts

- `run-id` is supplied on the command line and identifies one run for one
  case. Reusing it overwrites that run's files.
- The default training run is `train`; the default registration run is
  `register`.
- The imported active model for each case is the default training checkpoint:
  `checkpoints/<case>/train/best.ckpt`.
- A newly trained checkpoint must remain at
  `checkpoints/<case>/<training-run-id>/best.ckpt`.
- To register with a non-default training run, pass its `best.ckpt` explicitly
  using `--checkpoint`.

## Commands

Run all commands from `/home/zsr/project/vprn`:

```bash
# Convert only an explicitly selected CT DICOM series to the configured NIfTI path.
PYTHONPATH=src conda run -n mr2ct python scripts/convert_ct.py \
  --case sxh --source-dir /path/to/selected/ct/series

# Train; outputs go only to checkpoints/<case>/<run-id>/.
PYTHONPATH=src conda run -n mr2ct python scripts/train_case.py \
  --case sxh --run-id train

# Register; outputs go only to runs/<case>/<run-id>/.
PYTHONPATH=src conda run -n mr2ct python scripts/register_case.py \
  --case sxh --run-id register

# Register using a particular trained model.
PYTHONPATH=src conda run -n mr2ct python scripts/register_case.py \
  --case sxh --run-id register-custom \
  --checkpoint checkpoints/sxh/train-custom/best.ckpt

# Evaluate the matching registration run.
PYTHONPATH=src conda run -n mr2ct python scripts/evaluate_case.py \
  --case sxh --run-id register
```

Use `--dry-run` to check paths before a GPU run. Request permission before
running a command that requires GPU access.

## Code Boundaries

- `src/vprn/data/`: NIfTI CT loading, X-ray DICOM loading, and the unified
  Dataset. Preserve per-case geometry through `cases.json`.
- `src/vprn/models/registry.py`: selects and loads models. The network classes
  are currently read-only legacy imports from `diffpose/ours/utils/model2.py`.
- `src/vprn/rendering/`: DRR creation and shared image transforms. The DRR
  classes are currently read-only legacy imports.
- `src/vprn/training/`: reusable training loop, losses, and checkpoint format.
- `src/vprn/registration/`: model initialization plus pose-regression,
  sparse/gradient, and CMA-ES registration backends.
- `src/vprn/evaluation/`: result directories and metric aggregation.

Do not guess a case's CT series, axis transform, crop, zoom, fiducial
projection, pose parameterization, or transform direction. Inspect the
configured value and the relevant existing evidence first. If it is ambiguous,
ask the user before changing it.

## Git Rules

The following local assets are intentionally Git-ignored:

```text
/data/
/checkpoints/
```

Do not force-add the ignored data or checkpoint assets. Registration results
under `runs/` are not ignored and may be committed when they are the intended
deliverable. Python caches and pytest caches are also ignored.

## Verification

- Run focused tests after a narrow change; run the relevant test suite after
  shared Dataset, configuration, rendering, model registry, or output-layout
  changes.
- At minimum, run `PYTHONPATH=src conda run -n mr2ct pytest -q tests` when the
  complete suite is practical. If it stalls or cannot run, report exactly what
  completed and why the remainder did not.
- Before reporting success for a geometry or pose change, verify the affected
  case's data/pose behavior rather than relying only on a generic unit test.
