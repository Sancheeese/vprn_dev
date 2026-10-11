import pytest
import torch

from vprn.models.registry import load_checkpoint, model_target, supported_models


def test_model_registry_contains_active_model_variants():
    names = supported_models()
    assert "PoseRegressorAddSpDeco" in names
    assert "PoseRegressorCoeSpDeco" in names
    assert "PoseRegressorGSpDeco" in names


def test_model_registry_resolves_target_without_importing_model():
    module_name, class_name = model_target("PoseRegressorAddSpDeco")
    assert module_name == "ours.utils.model2"
    assert class_name == "PoseRegressorAddSpDeco"


def test_model_registry_rejects_unknown_model():
    with pytest.raises(KeyError, match="Unknown model"):
        model_target("MissingModel")


def test_load_checkpoint_accepts_training_checkpoint_schema(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "checkpoint.ckpt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 4}, path)

    restored = torch.nn.Linear(2, 1)
    metadata = load_checkpoint(restored, path, device="cpu")

    assert metadata["epoch"] == 4
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)
