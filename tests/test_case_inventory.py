from pathlib import Path

from vprn.config import load_cases
from vprn.models.registry import supported_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "configs" / "cases.json"


def test_inventory_contains_only_the_five_active_cases():
    cases = load_cases(CASES_PATH)

    assert set(cases) == {"ysy", "sxh", "wfl", "xyl", "zyl"}
    for case in cases.values():
        assert case.ct_path.is_file(), case.ct_path
        assert case.xray_path.is_dir(), case.xray_path
        assert case.seg_path.is_dir(), case.seg_path
        assert case.gt_pose_path.is_dir(), case.gt_pose_path
        assert case.checkpoint.is_file(), case.checkpoint
        assert case.checkpoint.parent.name == "train"
        assert case.model in supported_models()
        if case.train_model is not None:
            assert case.train_model in supported_models()
        if case.register_model is not None:
            assert case.register_model in supported_models()
        if case.train_checkpoint is not None:
            assert case.train_checkpoint.is_file(), case.train_checkpoint
        if case.register_checkpoint is not None:
            assert case.register_checkpoint.is_file(), case.register_checkpoint
        if case.fiducial_path is not None:
            assert case.fiducial_path.is_file(), case.fiducial_path


def test_inventory_has_provenance_for_each_converted_ct():
    cases = load_cases(CASES_PATH)

    for case in cases.values():
        provenance = case.ct_path.with_name(f"{case.ct_path.name}.json")
        assert provenance.is_file(), provenance
