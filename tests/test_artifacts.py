import json
from pathlib import Path

from vprn.evaluation.artifacts import ArtifactStore


def test_artifact_store_creates_isolated_run_layout_and_config_snapshot(tmp_path: Path):
    store = ArtifactStore(tmp_path / "runs", "sxh", "smoke-001")
    store.ensure()
    snapshot = store.save_config({"case": "sxh", "model": "PoseRegressorAddSpDeco"})

    assert snapshot == tmp_path / "runs" / "sxh" / "smoke-001" / "config.json"
    assert snapshot.is_file()
    assert json.loads(snapshot.read_text())["case"] == "sxh"
    assert store.registration_dir.is_dir()
    assert store.metrics_dir.is_dir()
    assert store.figures_dir.is_dir()


def test_artifact_store_writes_json_and_csv_in_run_directories(tmp_path: Path):
    store = ArtifactStore(tmp_path / "runs", "ysy", "run-1")

    json_path = store.write_json("metrics/summary.json", {"dice": 0.5})
    csv_path = store.write_csv("registration/poses.csv", [{"index": 0, "tx": 1.0}, {"index": 1, "tx": 2.0}])

    assert json.loads(json_path.read_text())["dice"] == 0.5
    assert csv_path.read_text().splitlines() == ["index,tx", "0,1.0", "1,2.0"]


def test_artifact_store_can_leave_checkpoints_out_of_registration_runs(tmp_path: Path):
    store = ArtifactStore(tmp_path / "runs", "sxh", "register-001", include_checkpoints=False)
    store.ensure()

    assert store.registration_dir.is_dir()
    assert not store.checkpoints_dir.exists()


def test_evaluation_filters_non_numeric_pose_metadata():
    from vprn.evaluation.metrics import numeric_metric_rows

    rows = [{"index": "0", "tx": "1.5", "method": "sparse"}]

    assert numeric_metric_rows(rows) == [{"tx": 1.5}]
