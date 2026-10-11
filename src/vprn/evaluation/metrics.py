from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def numeric_metric_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, float]]:
    """Keep numeric result columns while ignoring index/method metadata."""
    numeric_rows: list[dict[str, float]] = []
    for row in rows:
        numeric: dict[str, float] = {}
        for key, value in row.items():
            if key == "index":
                continue
            try:
                numeric[key] = float(value)
            except (TypeError, ValueError):
                continue
        numeric_rows.append(numeric)
    return numeric_rows


def mean_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            values.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in values.items() if items}
