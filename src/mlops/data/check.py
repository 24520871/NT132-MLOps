"""Command-line data quality gate used before training."""
import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlops.config import load_config
from mlops.data import ingest
from mlops.data.validation import DataValidationError, check_columns, validate
from usecases.churn import data as churn_data
from usecases.churn.schema import FEATURE_COLUMNS, ID_COLUMNS, SCHEMA, TARGET

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Machine-readable summary of the pre-training data checks."""

    passed: bool
    dataset: str
    rows: int | None
    columns: int | None
    checks: list[str]
    failures: list[str]


def _duplicate_headers(path: Path) -> list[str]:
    """Read the original CSV header because pandas renames duplicate headers."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    return sorted({name for name in header if header.count(name) > 1})


def validate_dataset(usecase: str) -> ValidationResult:
    """Validate the configured dataset using the use-case's existing loader/schema."""
    checks: list[str] = []
    dataset = "configured dataset"
    rows: int | None = None
    columns: int | None = None
    try:
        if usecase != "churn":
            raise ValueError(f"Data validation is implemented only for use case 'churn', got '{usecase}'.")

        cfg = load_config(usecase)
        dataset = Path(cfg["data"]["path"]).name
        checks.append("dataset path exists")
        path = ingest.locate(cfg["data"])
        dataset = path.name
        checks.append("CSV header has no duplicate column names")
        duplicates = _duplicate_headers(path)
        if duplicates:
            raise DataValidationError([f"duplicate CSV column names: {duplicates}"])

        checks.append("CSV is readable and has data rows")
        raw = ingest.read_csv(path)
        rows, columns = raw.shape
        if raw.empty:
            raise DataValidationError(["dataset contains no data rows"])

        checks.append("required identifier, feature, and target columns exist")
        check_columns(raw, ID_COLUMNS + FEATURE_COLUMNS + [TARGET])

        checks.append("nonblank TotalCharges values are numeric")
        charges = raw["TotalCharges"]
        charge_text = charges.fillna("").astype(str).str.strip()
        invalid_charges = charge_text.ne("") & pd.to_numeric(charge_text, errors="coerce").isna()
        if invalid_charges.any():
            raise DataValidationError(
                [f"'TotalCharges' has {int(invalid_charges.sum())} non-numeric non-blank values"]
            )

        frame = churn_data.clean(raw)
        checks.append("feature types, missing values, categories, and numeric ranges match schema")
        validate(frame, SCHEMA)

        checks.append("target labels are valid and contain both supervised classes")
        labels = set(frame[TARGET].unique())
        if labels != {0, 1}:
            raise DataValidationError(
                [f"target '{TARGET}' must contain both classes 0 and 1; found {sorted(labels)}"]
            )
        label_counts = frame[TARGET].value_counts()
        checks.append("row count and class counts support configured stratified training split")
        if label_counts.min() < 2:
            raise DataValidationError(
                ["each target class needs at least 2 rows for the configured stratified split"]
            )
        return ValidationResult(True, dataset, rows, columns, checks, [])
    except FileNotFoundError:
        message = (
            f"Dataset '{dataset}' was not found; place the configured churn CSV under data/raw "
            "or set DATA_PATH. No synthetic fallback is used."
        )
        logger.warning("Dataset validation failed for %s: %s", dataset, message)
        return ValidationResult(False, dataset, rows, columns, checks, [message])
    except (OSError, UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError,
            ValueError, KeyError) as exc:
        logger.warning("Dataset validation failed for %s: %s", dataset, exc)
        return ValidationResult(False, dataset, rows, columns, checks, [str(exc)])


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation and return a process status suitable for Airflow BashOperator."""
    parser = argparse.ArgumentParser(description="Validate a configured training dataset.")
    parser.add_argument("usecase", help="configured use case (currently: churn)")
    args = parser.parse_args(argv)
    result = validate_dataset(args.usecase)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
