"""Performance metrics for prediction records that have aligned ground-truth labels."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from mlops.serving.prediction_log import read_prediction_log

LABEL_COLUMNS = ("y_true", "ground_truth", "Churn")


@dataclass(frozen=True)
class PerformanceResult:
    recall: float | None
    auc: float | None
    sample_count: int
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.recall is not None or self.auc is not None


def _unavailable(reason: str, sample_count: int = 0) -> PerformanceResult:
    return PerformanceResult(None, None, sample_count, reason)


def _binary_labels(values: pd.Series) -> np.ndarray:
    if values.isna().any():
        raise ValueError("ground-truth labels contain missing values")
    if pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
        labels = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError("ground-truth labels must contain only 0/1 or Yes/No")
        return labels.astype(int)

    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"0": 0, "1": 1, "no": 0, "yes": 1}
    if not normalized.isin(mapping).all():
        raise ValueError("ground-truth labels must contain only 0/1 or Yes/No")
    return normalized.map(mapping).to_numpy(dtype=int)


def evaluate_performance(
    prediction_log: pd.DataFrame | str | Path,
) -> PerformanceResult:
    """Calculate recall/AUC only when labels are embedded in the prediction records.

    The serving log has no request ID or ground-truth field by default. Separate labels
    cannot safely be aligned to it by row order, so missing labels remain unavailable.
    """
    records = (
        read_prediction_log(prediction_log)
        if isinstance(prediction_log, (str, Path))
        else prediction_log
    )
    if not isinstance(records, pd.DataFrame):
        raise TypeError("prediction_log must be a DataFrame or a path to a JSONL prediction log")
    if records.empty:
        return _unavailable("prediction log is empty")

    label_columns = [col for col in LABEL_COLUMNS if col in records.columns]
    if len(label_columns) > 1:
        raise ValueError(f"prediction log has multiple ground-truth columns: {label_columns}")
    if not label_columns:
        return _unavailable(
            "ground-truth labels are unavailable; prediction logs do not include labels"
        )

    required = ("prediction", "probability")
    missing = [col for col in required if col not in records.columns]
    if missing:
        raise ValueError(f"prediction log is missing required columns: {missing}")

    labels = _binary_labels(records[label_columns[0]])
    if records["prediction"].isna().any() or records["probability"].isna().any():
        raise ValueError("prediction log contains missing predictions or probabilities")
    try:
        predictions = pd.to_numeric(records["prediction"], errors="raise").to_numpy(dtype=float)
        probabilities = pd.to_numeric(records["probability"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("predictions and probabilities must be numeric") from exc

    if not np.isfinite(predictions).all() or not np.isin(predictions, [0.0, 1.0]).all():
        raise ValueError("predictions must contain only 0 or 1")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("probabilities must be finite values between 0 and 1")
    if len(labels) != len(predictions):
        raise ValueError("prediction and ground-truth row counts do not match")

    classes = np.unique(labels)
    recall = (
        float(recall_score(labels, predictions, zero_division=0))
        if 1 in classes
        else None
    )
    auc = (
        float(roc_auc_score(labels, probabilities))
        if len(classes) == 2
        else None
    )
    reasons = []
    if recall is None:
        reasons.append("recall unavailable: ground truth contains no positive class")
    if auc is None:
        reasons.append("AUC unavailable: ground truth must contain both classes")
    return PerformanceResult(
        recall=recall,
        auc=auc,
        sample_count=len(labels),
        reason="; ".join(reasons) or None,
    )
