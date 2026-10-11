from pathlib import Path
from types import SimpleNamespace

from vprn.data.convert_ct import convert_ct_series


def test_convert_ct_series_uses_ct_modality_and_copies_only_selected_series(tmp_path):
    source_dir = tmp_path / "source" / "CT" / "3"
    source_dir.mkdir(parents=True)
    destination = tmp_path / "data" / "sxh" / "ct" / "ct.nii"
    calls = []

    def fake_convert_study(study, output_root, overwrite, min_dicom_files):
        calls.append((study, output_root, overwrite, min_dicom_files))
        generated = output_root / "孙新华" / "CT" / "3.nii"
        generated.parent.mkdir(parents=True)
        generated.write_bytes(b"nifti")
        return {"status": "converted", "nii_path": generated, "selected_count": 10, "dropped_count": 0}

    converter = SimpleNamespace(
        Study=lambda patient_name, modality, series_dir: (patient_name, modality, series_dir),
        convert_study=fake_convert_study,
    )

    result = convert_ct_series(
        source_dir=source_dir,
        destination=destination,
        patient_name="孙新华",
        series_name="3",
        converter=converter,
    )

    assert result.status == "converted"
    assert result.selected_count == 10
    assert destination.read_bytes() == b"nifti"
    assert calls[0][0][1] == "CT"
    assert calls[0][0][2] == source_dir


def test_convert_ct_series_does_not_overwrite_without_flag(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    destination = tmp_path / "ct.nii"
    destination.write_bytes(b"existing")

    result = convert_ct_series(
        source_dir=source_dir,
        destination=destination,
        patient_name="Demo",
        series_name="1",
        converter=SimpleNamespace(),
    )

    assert result.status == "skipped"
    assert destination.read_bytes() == b"existing"

