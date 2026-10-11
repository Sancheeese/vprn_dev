from pathlib import Path

from vprn.cli.common import checkpoint_run_dir


def test_checkpoint_run_directory_is_case_and_run_scoped(tmp_path: Path):
    assert checkpoint_run_dir(tmp_path / "checkpoints", "sxh", "train-001") == (
        tmp_path / "checkpoints" / "sxh" / "train-001"
    )
