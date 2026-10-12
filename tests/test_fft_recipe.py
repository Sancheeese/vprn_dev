import importlib
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vprn.config import load_case
from vprn.rendering.drr import _prepare_legacy_imports
from vprn.training.fft_recipe import (
    build_fft_dataset, fft_case, fft_scheduler, fft_transforms,
    prepare_fft_batch, random_fft_offset,
)
from vprn.training.trainer import Trainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("use_tube", [False, True])
def test_fft_inputs_match_original_trainer_with_fixed_randomness(use_tube):
    _prepare_legacy_imports()
    from ours.fft import high_pass_gaussian
    from ours.utils.CT_dataset import Transforms
    from ours.utils.mask_data import MaskDataAugmentation

    torch.manual_seed(3)
    image = torch.rand(2, 1, 256, 256)
    bone = torch.rand_like(image)
    augmenter = MaskDataAugmentation(PROJECT_ROOT.parent / "diffpose/ours/tube") if use_tube else None
    transforms = Transforms(256, radius=119)
    # The reference sequence is copied from train_ysy_addspdeco_fft.py.
    target_bone = transforms(bone.float(), reverse=False)
    target_bone = (target_bone - target_bone.min()) / (target_bone.max() - target_bone.min())
    target_bone = torch.tanh(50 * target_bone)
    target_image = transforms(image).float()
    expected_input = transforms(high_pass_gaussian(target_image, 0.2), reverse=False).float()
    random.seed(12)
    if use_tube:
        random.random()
        expected_input, tube = augmenter.augment(expected_input)
        target_bone = target_bone * tube
    expected_input = transforms(expected_input, reverse=False).float()
    random.seed(12)
    actual_image, actual_input, actual_bone, tube_applied = prepare_fft_batch(
        image, bone, fft_transforms(256), 0.2, augmenter, 1.0,
    )
    torch.testing.assert_close(actual_image, target_image, rtol=0, atol=0)
    torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
    torch.testing.assert_close(actual_bone, target_bone, rtol=0, atol=0)
    assert tube_applied == use_tube


def test_random_fft_pose_matches_original_sampler():
    _prepare_legacy_imports()
    from ours.case.my_util2 import get_random_offset

    torch.manual_seed(123)
    expected = get_random_offset(8, "cpu").get_matrix()
    torch.manual_seed(123)
    actual = random_fft_offset(8, "cpu").get_matrix()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fft_learning_rate_matches_original_at_warmup_and_cosine_boundaries():
    from pytorch_transformers.optimization import WarmupCosineSchedule

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=0.001)
    scheduler = fft_scheduler(optimizer, 1000, 100, 500)
    reference_optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(()))], lr=0.001)
    reference = WarmupCosineSchedule(reference_optimizer, 500, 99500)
    for step in (0, 1, 499, 500, 501, 50000, 99500, 100000):
        assert scheduler.lr_lambdas[0](step) == reference.lr_lambdas[0](step)


def test_fft_optimizer_update_matches_original_agc_and_accumulation():
    from timm.utils.agc import adaptive_clip_grad

    model = torch.nn.Linear(2, 1)
    reference = torch.nn.Linear(2, 1)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    batches = [torch.ones(1, 2) * 100, torch.ones(1, 2) * 200]
    def loss(current, batch):
        return (current(batch) - 10).square().mean()
    for batch in batches:
        loss(reference, batch).backward()
    adaptive_clip_grad(reference.parameters())
    reference_optimizer.step()
    Trainer(model, optimizer, loss, accumulate_steps=2,
            gradient_clip_fn=adaptive_clip_grad, flush_accumulation=False).fit(batches, epochs=1)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("case_id,module_name", [
    ("ysy", "ours.case.ysy.CT_dataset"),
    ("sxh", "ours.case.sxh.CT_dataset_iso"),
    ("zyl", "ours.utils.CT_dataset_PA"),
])
def test_fft_ct_and_centred_poses_match_original_dicom_loader(case_id, module_name):
    case = fft_case(load_case(case_id, PROJECT_ROOT / "configs/cases.json"))
    actual = build_fft_dataset(case)
    provenance = json.loads(case.ct_path.with_name("ct.nii.json").read_text())
    source = PROJECT_ROOT.parent / provenance["source_series_dir"]
    _prepare_legacy_imports()
    loader = importlib.import_module(module_name).IntubationDataset
    geometry = case.geometry
    expected = loader(str(source), str(case.xray_path), x_offset=geometry.x_offset,
                      y_offset=geometry.y_offset, z_offset=geometry.z_offset,
                      z_cut=geometry.z_cut, factors=list(geometry.factors))
    assert actual.volume.shape == expected.volume.shape
    np.testing.assert_allclose(actual.spacing, expected.spacing, rtol=2e-6)
    np.testing.assert_allclose(actual.volume[::13, ::13, ::13],
                               expected.volume[::13, ::13, ::13], rtol=0, atol=1e-4)
    for name in ("isocenter_pose", "center_pose", "back_pose"):
        torch.testing.assert_close(getattr(actual, name).get_matrix(),
                                   getattr(expected, name).get_matrix(), rtol=0, atol=1e-4)


def test_zyl_fft_selects_addspdeco_without_replacing_registration_model():
    original = load_case("zyl", PROJECT_ROOT / "configs/cases.json")
    case = fft_case(original)
    assert case.train_model == "PoseRegressorAddSpDeco"
    assert original.train_model == case.register_model == "PoseRegressorCoeSpDeco"
    assert case.geometry.factors == original.geometry.factors
    assert original.geometry.post_flip_axes == (0,)
    assert case.geometry.post_flip_axes == ()


def test_fft_entrypoint_resume_matches_continuous_updates(monkeypatch, tmp_path):
    """Exercise checkpoint/epoch/LR wiring with cheap deterministic renderers."""
    spec = importlib.util.spec_from_file_location("fft_entrypoint", PROJECT_ROOT / "scripts/train_case_fft.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    from vprn.data.dataset import _pose, _rigid_transform_class

    rigid = _rigid_transform_class()
    identity = _pose(rigid, [0, 0, 0], [0, 0, 0])
    dataset = SimpleNamespace(isocenter_pose=identity, center_pose=identity,
                              back_pose=identity, sdr=500.0, delx=1.0)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.7))

        def forward(self, image, mask):
            return identity, self.weight.expand_as(mask)

    rendered = torch.linspace(0, 1, 16 * 16).reshape(1, 1, 16, 16)
    monkeypatch.setattr(script, "build_fft_dataset", lambda case: dataset)
    monkeypatch.setattr(script, "build_model", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(script, "create_drr", lambda *args, **kwargs: lambda *a, **k: rendered)
    monkeypatch.setattr(script, "fft_transforms", lambda height: lambda image, **kw: image)
    monkeypatch.setattr(script, "random_fft_offset", lambda *args: identity)
    monkeypatch.setattr(script, "registration_loss", lambda *args, **kw: kw["predicted_mask"].square().mean())
    monkeypatch.setattr(script, "checkpoint_run_dir", lambda root, case, run: tmp_path / run)

    def run(run_id, resume=None):
        monkeypatch.setattr("sys.argv", [str(spec.origin), "--case", "zyl", "--device", "cpu",
                            "--run-id", run_id, "--epochs", "12", "--batches-per-epoch", "2",
                            "--batch-size", "1", "--height", "16", "--tube-augmentation-prob", "0",
                            "--save-every-epochs", "1"] + (["--resume", str(resume)] if resume else []))
        assert script.main() == 0
        return torch.load(tmp_path / run_id / "last.ckpt", weights_only=False)

    continuous = run("continuous")
    checkpoint = tmp_path / "continuous/epoch0006.ckpt"
    resumed = run("resumed", checkpoint)
    torch.testing.assert_close(resumed["model_state_dict"]["weight"],
                               continuous["model_state_dict"]["weight"], rtol=0, atol=0)
    assert resumed["optimizer_state_dict"]["param_groups"] == continuous["optimizer_state_dict"]["param_groups"]
    assert resumed["scheduler_state_dict"] == continuous["scheduler_state_dict"]
    assert resumed["epoch"] == 12
    assert resumed["model_class"] == "PoseRegressorAddSpDeco"
    assert resumed["model_name"] == "resnet18"
    history = json.loads((tmp_path / "resumed/train_history.json").read_text())
    assert [row["epoch"] for row in history] == list(range(7, 13))


def test_fft_all_nan_epoch_fails_before_checkpoint_callback():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, optimizer, lambda *args: torch.tensor(float("nan")), skip_nan_loss=True)
    with pytest.raises(FloatingPointError, match="no valid losses"):
        trainer.fit([None], epochs=1, on_epoch_end=lambda *args: pytest.fail("saved invalid checkpoint"))
