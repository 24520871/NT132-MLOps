"""Generic schema validation, run before any data reaches training."""
from typing import Any

import pandas as pd


class DataValidationError(ValueError):
    """Raised when a dataset does not match its schema; message lists every problem found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Data validation failed:\n- " + "\n- ".join(problems))


def check_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError([f"missing columns: {missing}"])


def validate(df: pd.DataFrame, schema: dict[str, Any]) -> None:
    """Check row count, columns, nulls, numeric dtype/range and allowed categories."""
    problems: list[str] = []

    if len(df) < schema.get("min_rows", 1):
        problems.append(f"only {len(df)} rows, expected at least {schema.get('min_rows', 1)}")

    for col, spec in schema["columns"].items():
        if col not in df.columns:
            problems.append(f"missing column '{col}'")
            continue
        series = df[col]
        n_null = int(series.isna().sum())
        if n_null:
            problems.append(f"'{col}' has {n_null} missing values")
        values = series.dropna()

        if spec["kind"] == "numeric":
            if not pd.api.types.is_numeric_dtype(values):
                problems.append(f"'{col}' must be numeric, got dtype {series.dtype}")
                continue
            if "min" in spec and (values < spec["min"]).any():
                problems.append(f"'{col}' has values below {spec['min']} (min {values.min()})")
            if "max" in spec and (values > spec["max"]).any():
                problems.append(f"'{col}' has values above {spec['max']} (max {values.max()})")
        else:
            bad = sorted({str(v) for v in values.unique() if v not in spec["allowed"]})
            if bad:
                problems.append(f"'{col}' has unexpected values {bad}; allowed {spec['allowed']}")

    if problems:
        raise DataValidationError(problems)
