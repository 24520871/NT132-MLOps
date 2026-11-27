"""Objective Gate 2 checks for models prepared in the registry's staging alias."""
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd


class ProbabilityModel(Protocol):
    """The fitted churn pipelines expose class probabilities through this method."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class Gate2Decision:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    observations: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"{status}: " + "; ".join(self.reasons)


def _signature_problem(signature_fields: list[str] | None, expected: list[str], label: str) -> str | None:
    if signature_fields is None:
        return f"{label} model has no MLflow input signature"
    if len(signature_fields) != len(set(signature_fields)):
        return f"{label} model input signature contains duplicate feature names"
    if set(signature_fields) != set(expected):
        missing = sorted(set(expected) - set(signature_fields))
        unexpected = sorted(set(signature_fields) - set(expected))
        return f"{label} model input signature mismatch; missing={missing}, unexpected={unexpected}"
    return None


def _predict(
    model: ProbabilityModel,
    features: pd.DataFrame,
    label: str,
) -> tuple[np.ndarray | None, float, str | None]:
    started = perf_counter()
    try:
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
    except Exception as exc:
        elapsed_ms = (perf_counter() - started) * 1000
        return None, elapsed_ms, f"{label} model prediction failed: {exc}"
    elapsed_ms = (perf_counter() - started) * 1000

    if probabilities.shape != (len(features), 2):
        return None, elapsed_ms, (
            f"{label} model returned shape {probabilities.shape}; expected ({len(features)}, 2)"
        )
    if not np.isfinite(probabilities).all():
        return None, elapsed_ms, f"{label} model returned non-finite probabilities"
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        return None, elapsed_ms, f"{label} model returned probabilities outside [0, 1]"
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
        return None, elapsed_ms, f"{label} model probabilities do not sum to 1"
    return probabilities[:, 1], elapsed_ms, None


def evaluate_gate2(
    candidate_model: ProbabilityModel,
    production_model: ProbabilityModel,
    features: pd.DataFrame,
    expected_features: list[str],
    candidate_signature: list[str] | None,
    production_signature: list[str] | None,
    threshold: float,
) -> Gate2Decision:
    """Validate signature and predictions; report model comparison and latency as observations.

    No undocumented positive-rate, agreement, or latency threshold is applied. A production model
    is required by the caller and is evaluated on the exact same held-out feature rows.
    """
    reasons = [
        problem
        for problem in (
            _signature_problem(candidate_signature, expected_features, "staging"),
            _signature_problem(production_signature, expected_features, "production"),
        )
        if problem is not None
    ]
    if list(features.columns) != expected_features:
        reasons.append("held-out feature columns do not match the expected churn schema and order")
    if features.empty:
        reasons.append("held-out sample is empty")
    if not 0.0 <= threshold <= 1.0:
        reasons.append(f"prediction threshold must be between 0 and 1, got {threshold}")
    if reasons:
        return Gate2Decision(False, reasons)

    candidate_probabilities, candidate_ms, candidate_error = _predict(
        candidate_model, features, "staging"
    )
    production_probabilities, production_ms, production_error = _predict(
        production_model, features, "production"
    )
    errors = [error for error in (candidate_error, production_error) if error is not None]
    if errors or candidate_probabilities is None or production_probabilities is None:
        return Gate2Decision(False, errors or ["model prediction output was unavailable"])

    candidate_labels = candidate_probabilities >= threshold
    production_labels = production_probabilities >= threshold
    observations = {
        "sample_count": float(len(features)),
        "staging_positive_rate": float(candidate_labels.mean()),
        "production_positive_rate": float(production_labels.mean()),
        "prediction_agreement_rate": float((candidate_labels == production_labels).mean()),
        "mean_absolute_probability_difference": float(
            np.abs(candidate_probabilities - production_probabilities).mean()
        ),
        "staging_latency_ms_per_row": candidate_ms / len(features),
        "production_latency_ms_per_row": production_ms / len(features),
    }
    return Gate2Decision(
        True,
        [
            "both signatures match the expected feature schema",
            "both models loaded and returned finite, valid class probabilities",
            "prediction comparison and inference latency are informational only",
        ],
        observations,
    )
