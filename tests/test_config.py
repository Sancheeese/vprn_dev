import json
from pathlib import Path

import pytest

from vprn.config import CaseConfig, load_case, load_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_cases_file(tmp_path: Path) -> Path:
    config_path = tmp_path / "configs" / "cases.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "cases": {
                    "demo": {
                        "patient_name": "Demo Patient",
                        "ct_series": "3",
                        "ct_path": "data/demo/ct/ct.nii",
                        "xray_path": "data/demo/xray",
                        "seg_path": "data/demo/seg",
                        "gt_pose_path": "data/demo/gt_pose",
                        "model": "PoseRegressorAddSpDeco",
                        "checkpoint": "checkpoints/demo/best.ckpt",
                        "geometry": {
                            "x_offset": 20,
                            "y_offset": 30,
                            "z_offset": 40,
                            "z_cut": 100,
                            "z_cut_end": -1,
                            "factors": [0.5, 1.0, 1.5],
                            "axis_transform": "legacy_demo",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_load_case_resolves_paths_relative_to_project_root(tmp_path):
    config_path = write_cases_file(tmp_path)

    case = load_case("demo", config_path)

    assert isinstance(case, CaseConfig)
    assert case.case_id == "demo"
    assert case.ct_series == "3"
    assert case.ct_path == tmp_path / "data/demo/ct/ct.nii"
    assert case.xray_path == tmp_path / "data/demo/xray"
    assert case.geometry.factors == (0.5, 1.0, 1.5)
    assert case.geometry.axis_transform == "legacy_demo"


def test_load_cases_preserves_case_specific_models_and_geometry(tmp_path):
    config_path = write_cases_file(tmp_path)

    cases = load_cases(config_path)

    assert tuple(cases) == ("demo",)
    assert cases["demo"].model == "PoseRegressorAddSpDeco"
    assert cases["demo"].geometry.x_offset == 20


def test_load_case_preserves_role_specific_models_and_fiducial_projection(tmp_path):
    config_path = write_cases_file(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["cases"]["demo"].update(
        {
            "train_model": "PoseRegressorGSpDeco",
            "register_model": "PoseRegressorCoeSpDeco",
            "fiducial_projection": "swap_xy",
            "registration_params": {"rotation_lr": 0.01},
        }
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    case = load_case("demo", config_path)

    assert case.train_model == "PoseRegressorGSpDeco"
    assert case.register_model == "PoseRegressorCoeSpDeco"
    assert case.fiducial_projection == "swap_xy"
    assert case.registration_params == {"rotation_lr": 0.01}


def test_load_case_rejects_unknown_case(tmp_path):
    config_path = write_cases_file(tmp_path)

    with pytest.raises(KeyError, match="Unknown case: missing"):
        load_case("missing", config_path)


def test_load_case_rejects_unknown_registration_method(tmp_path):
    config_path = write_cases_file(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["cases"]["demo"]["registration_method"] = "not-a-backend"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="registration_method"):
        load_case("demo", config_path)


def test_load_case_rejects_invalid_spacing_permutation(tmp_path):
    config_path = write_cases_file(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["cases"]["demo"]["geometry"]["spacing_permutation"] = [0, 1]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="spacing_permutation"):
        load_case("demo", config_path)


def test_active_cases_preserve_legacy_bone_thresholds():
    cases = load_cases(PROJECT_ROOT / "configs" / "cases.json")

    assert {
        case_id: case.training_params["bone_threshold"]
        for case_id, case in cases.items()
    } == {"ysy": 250, "sxh": 300, "wfl": 350, "xyl": 350, "zyl": 350}
