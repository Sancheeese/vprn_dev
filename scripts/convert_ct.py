#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from vprn.config import load_case
from vprn.data.convert_ct import convert_ct_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one selected CT DICOM series to NIfTI.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cases.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = load_case(args.case, args.config)
    result = convert_ct_series(
        args.source_dir,
        case.ct_path,
        case.patient_name,
        case.ct_series,
        overwrite=args.overwrite,
    )
    print(f"[{result.status}] {case.case_id}: {result.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
