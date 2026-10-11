import torch
import numpy as np


def test_bone_drr_factory_initializes_density_for_sparse_registration():
    from types import SimpleNamespace

    from vprn.rendering.drr import create_drr

    dataset = SimpleNamespace(
        volume=np.array([[[-1000.0, 0.0], [500.0, 0.0]], [[0.0, 500.0], [-1000.0, 0.0]]], dtype=np.float32),
        spacing=np.array([1.0, 1.0, 1.0]),
        sdr=500.0,
        delx=1.0,
    )

    drr_bone = create_drr(dataset, height=4, device="cpu", bone=True)

    assert hasattr(drr_bone, "density")


def test_bone_drr_factory_accepts_a_case_specific_threshold():
    from types import SimpleNamespace

    from vprn.rendering.drr import create_drr

    dataset = SimpleNamespace(
        volume=np.array([[[300.0, 350.0, 351.0]]], dtype=np.float32),
        spacing=np.array([1.0, 1.0, 1.0]),
        sdr=500.0,
        delx=1.0,
    )

    drr_bone = create_drr(dataset, height=4, device="cpu", bone=True, bone_threshold=350)

    assert drr_bone.bone.flatten().tolist() == [False, False, True]

from vprn.rendering.transforms import ImageTransforms


def test_image_transforms_resize_normalize_and_apply_circle_mask():
    transform = ImageTransforms(size=4, radius=1)
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    result = transform(image, reverse=False)

    assert result.shape == (1, 1, 4, 4)
    assert torch.all(result[..., 0, 0] == result.min())
    assert torch.isfinite(result).all()


def test_image_transforms_accepts_numpy_xray_pixels():
    transform = ImageTransforms(size=4, radius=1)

    result = transform(np.arange(16, dtype=np.float32).reshape(4, 4), reverse=True)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 1, 4, 4)
