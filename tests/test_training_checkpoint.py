from pathlib import Path

import torch

from vprn.training.checkpoint import save_checkpoint


def test_save_checkpoint_writes_model_optimizer_and_metadata(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "run" / "checkpoint.ckpt"

    save_checkpoint(path, model, optimizer, epoch=3, loss=0.25)

    payload = torch.load(path, map_location="cpu")
    assert payload["epoch"] == 3
    assert payload["loss"] == 0.25
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload

