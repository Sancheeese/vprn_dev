from pathlib import Path

import numpy as np

from vprn.config import load_case, load_cases
from vprn.data.dataset import UnifiedDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "configs" / "cases.json"


def test_all_cases_have_expected_legacy_volume_geometry():
    expected = {
        "ysy": ((410, 123, 320), (0.99340176, 2.0, 0.99365234)),
        "sxh": ((307, 307, 321), (1.24348958, 1.24348958, 1.33333333)),
        "wfl": ((256, 256, 220), (1.46484399, 1.46484399, 1.25)),
        "xyl": ((256, 292, 294), (1.26190476, 0.75, 1.26171875)),
        "zyl": ((232, 309, 260), (1.5, 1.12694301, 1.12694301)),
    }
    cases = load_cases(CASES_PATH)

    for case_id, (shape, spacing) in expected.items():
        dataset = UnifiedDataset(cases[case_id])
        assert dataset.volume.shape == shape
        np.testing.assert_allclose(dataset.spacing, spacing, rtol=1e-5, atol=1e-5)

