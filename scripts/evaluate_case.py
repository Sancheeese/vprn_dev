#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vprn.evaluation.artifacts import ArtifactStore  # noqa: E402
from vprn.evaluation.metrics import mean_metrics, numeric_metric_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one registration run.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = ArtifactStore(args.runs_root, args.case, args.run_id, include_checkpoints=False)
    pose_path = store.registration_dir / "poses.csv"
    if not pose_path.is_file():
        raise FileNotFoundError(f"registration result not found: {pose_path}")
    with pose_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_rows = numeric_metric_rows(rows)
    summary = mean_metrics(numeric_rows)
    output = store.write_json("metrics/summary.json", {"count": len(rows), "mean": summary})
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
