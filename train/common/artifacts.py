from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def write_csv_rows(
    rows: Sequence[dict[str, object]],
    fieldnames: Sequence[str],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def build_dense_metric_matrix(
    rows: Sequence[dict[str, object]],
    *,
    row_key: str,
    column_key: str,
    value_key: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    ordered_rows = sorted({int(row[row_key]) for row in rows})
    ordered_columns = sorted({int(row[column_key]) for row in rows})
    row_to_idx = {value: idx for idx, value in enumerate(ordered_rows)}
    column_to_idx = {value: idx for idx, value in enumerate(ordered_columns)}
    matrix = np.full((len(ordered_rows), len(ordered_columns)), np.nan, dtype=np.float64)
    for row in rows:
        matrix[row_to_idx[int(row[row_key])], column_to_idx[int(row[column_key])]] = float(
            row[value_key]
        )
    if np.isnan(matrix).any():
        raise ValueError("Rows do not form a complete dense matrix.")
    return matrix, ordered_rows, ordered_columns
