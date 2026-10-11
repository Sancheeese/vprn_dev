from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from vprn.config import GeometryConfig
from vprn.data.dataset import load_nifti_volume
from vprn.data.xray import XRaySeries


def test_unified_dataset_exposes_fiducial_projection_and_tre_for_active_case():
    from pathlib import Path

    from vprn.config import load_case
    from vprn.data.dataset import UnifiedDataset

    project = Path(__file__).resolve().parents[1]
    case = load_case("sxh", project / "configs" / "cases.json")
    dataset = UnifiedDataset(case)
    pose = dataset.get_manual_gt(0)

    true_2d, predicted_2d = dataset.get_2d_fiducials(0, pose)
    true_3d, predicted_3d = dataset.get_3d_fiducials(0, pose)

    assert true_2d.shape[-1] == 2
    assert true_3d.shape[-1] == 3
    assert true_2d.shape == predicted_2d.shape
    assert true_3d.shape == predicted_3d.shape
    assert float(dataset.calc_tre(0, pose)) == 0.0


def test_load_nifti_volume_applies_case_transform_crop_and_resampling(tmp_path: Path):
    data = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    nii_path = tmp_path / "ct.nii"
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), nii_path)

    geometry = GeometryConfig(
        z_cut=1,
        z_cut_end=3,
        factors=(0.5, 1.0, 1.0),
        axis_transform="transpose_0_1",
        post_flip_axes=(),
    )
    volume, spacing, header_affine = load_nifti_volume(nii_path, geometry)

    expected = np.transpose(data, (1, 0, 2))[:, :, 1:3]
    assert volume.shape == (2, 4, 2)
    np.testing.assert_allclose(volume, zoom(expected, (0.5, 1.0, 1.0), order=1))
    np.testing.assert_allclose(spacing, [6.0, 2.0, 4.0])
    np.testing.assert_allclose(header_affine, affine)


def test_xray_series_sorts_numeric_suffix_and_reads_pixel_arrays(tmp_path: Path):
    for name in ("frame_10.dcm", "frame_2.dcm", "frame_1.dcm"):
        (tmp_path / name).write_bytes(b"placeholder")

    class FakeDataset:
        def __init__(self, value):
            self.pixel_array = np.full((2, 3), value, dtype=np.uint16)

        def get(self, key, default=None):
            return {"DistanceSourceToDetector": 1000.0, "IntensifierSize": 400.0}.get(key, default)

    series = XRaySeries(tmp_path, read_fn=lambda path: FakeDataset(int(path.stem.split("_")[-1])))

    assert series.files == [tmp_path / "frame_1.dcm", tmp_path / "frame_2.dcm", tmp_path / "frame_10.dcm"]
    np.testing.assert_array_equal(series.read_pixels(1), np.full((2, 3), 2, dtype=np.uint16))
    assert series.metadata["DistanceSourceToDetector"] == 1000.0
