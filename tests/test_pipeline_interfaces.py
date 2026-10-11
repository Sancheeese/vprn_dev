from dataclasses import dataclass

import torch

from vprn.registration.engine import RegistrationEngine
from vprn.registration.initialization import predict_pose
from vprn.registration.optimizers import create_pose_optimizer, create_scheduler
from vprn.registration.backends import refine_pose
from vprn.training.trainer import Trainer
from vprn.training.losses import registration_loss


class TinyModel(torch.nn.Module):
    def forward(self, image, mask=None):
        del mask
        return image.mean(dim=tuple(range(1, image.ndim)))


@dataclass
class Batch:
    image: torch.Tensor
    target: torch.Tensor


def test_trainer_runs_a_configured_step_and_returns_history():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def step_fn(model, batch):
        prediction = model(batch.image)
        return torch.nn.functional.mse_loss(prediction, batch.target)

    batches = [Batch(torch.ones(1, 2), torch.zeros(1, 1))]
    history = Trainer(model, optimizer, step_fn).fit(batches, epochs=2)

    assert len(history) == 2
    assert set(history[0]) == {"epoch", "loss"}


def test_trainer_supports_gradient_accumulation():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    def step_fn(model, batch):
        return (model(batch) - 1.0).square().mean()

    Trainer(model, optimizer, step_fn, accumulate_steps=2).fit(
        [torch.ones(1, 1), torch.ones(1, 1)], epochs=1
    )

    assert torch.isfinite(model.weight).all()


def test_trainer_calls_epoch_end_callback_with_accumulated_history():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    received = []

    def step_fn(model, batch):
        return model(batch).square().mean()

    def on_epoch_end(epoch, record, history):
        received.append((epoch, record, len(history)))

    Trainer(model, optimizer, step_fn).fit(
        [torch.ones(1, 1)], epochs=2, on_epoch_end=on_epoch_end
    )

    assert [epoch for epoch, _record, _history_size in received] == [0, 1]
    assert [history_size for _epoch, _record, history_size in received] == [1, 2]


def test_registration_engine_supports_model_output_and_result_record():
    class Dataset:
        def __getitem__(self, index):
            return torch.ones(1, 1, 2, 2), {"index": index}

    class Model:
        def eval(self):
            return self

        def __call__(self, image, mask=None):
            del mask
            return image.mean(), None

    engine = RegistrationEngine(Model(), Dataset(), device="cpu")
    result = engine.run(index=3)

    assert result.index == 3
    assert result.prediction == 1.0
    assert result.target == {"index": 3}


def test_predict_pose_composes_model_offset_with_dataset_isocenter():
    from vprn.data.dataset import _rigid_transform_class

    rigid_transform = _rigid_transform_class()
    isocenter = rigid_transform(
        torch.eye(3), torch.tensor([10.0, 20.0, 30.0]),
    )

    class Dataset:
        isocenter_pose = isocenter

    class Model:
        def eval(self):
            return self

        def __call__(self, image, mask=None):
            del image, mask
            return rigid_transform(torch.eye(3), torch.tensor([1.0, 2.0, 3.0])), torch.zeros(1)

    prediction = predict_pose(Model(), torch.zeros(1, 1, 4, 4), Dataset())

    torch.testing.assert_close(prediction.get_translation(), torch.tensor([[11.0, 22.0, 33.0]]))


def test_create_pose_optimizer_keeps_separate_rotation_and_translation_rates():
    registration = torch.nn.Module()
    registration.rotation = torch.nn.Parameter(torch.zeros(1, 3))
    registration.translation = torch.nn.Parameter(torch.zeros(1, 3))

    optimizer = create_pose_optimizer(registration, rotation_lr=0.01, translation_lr=2.5)

    assert [group["lr"] for group in optimizer.param_groups] == [0.01, 2.5]


def test_create_scheduler_supports_step_and_none_modes():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)

    scheduler = create_scheduler(optimizer, kind="step", step_size=2, gamma=0.5)

    assert scheduler is not None
    assert create_scheduler(optimizer, kind="none") is None


def test_refine_pose_dispatch_returns_initial_pose_for_zero_iterations():
    class Drr:
        pass

    pose = object()
    assert refine_pose(pose, torch.zeros(1), Drr(), None, method="sparse", iterations=0) is pose
    assert refine_pose(pose, torch.zeros(1), Drr(), None, method="cmaes", iterations=0) is pose


def test_registration_loss_combines_ncc_pose_and_optional_mask_terms():
    class Metric:
        def __call__(self, predicted, target):
            del predicted, target
            return torch.tensor([0.5])

    class Geodesic:
        def __call__(self, predicted, target):
            del predicted, target
            return torch.tensor([0.2])

    class Double:
        def __call__(self, predicted, target):
            del predicted, target
            return torch.tensor([0.3]), torch.tensor([0.4]), torch.tensor([0.5])

    loss = registration_loss(
        object(), object(), torch.zeros(1), torch.zeros(1),
        predicted_mask=torch.ones(1, 1, 2, 2), target_mask=torch.zeros(1, 1, 2, 2),
        metric=Metric(), geodesic=Geodesic(), double_geodesic=Double(), mask_weight=10.0,
    )

    torch.testing.assert_close(loss, torch.tensor(10.507))
