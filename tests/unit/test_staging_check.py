import numpy as np
import pandas as pd
import pytest

from mlops.training.staging_check import evaluate_gate2


class ProbabilityModel:
    def __init__(self, probabilities=None, error=None):
        self.probabilities = probabilities
        self.error = error

    def predict_proba(self, features):
        if self.error:
            raise self.error
        return self.probabilities


def _evaluate(candidate, production, candidate_signature=None, production_signature=None):
    features = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": ["a", "b"]})
    signature = ["feature_a", "feature_b"]
    return evaluate_gate2(
        candidate,
        production,
        features,
        signature,
        signature if candidate_signature is None else candidate_signature,
        signature if production_signature is None else production_signature,
        0.5,
    )


def test_gate2_passes_objective_checks_and_reports_comparison_and_latency():
    candidate = ProbabilityModel(np.array([[0.2, 0.8], [0.7, 0.3]]))
    production = ProbabilityModel(np.array([[0.3, 0.7], [0.6, 0.4]]))

    result = _evaluate(candidate, production)

    assert result.passed
    assert result.observations["sample_count"] == 2
    assert result.observations["prediction_agreement_rate"] == 1.0
    assert result.observations["mean_absolute_probability_difference"] == pytest.approx(0.1)
    assert result.observations["staging_latency_ms_per_row"] >= 0.0
    assert "informational only" in result.summary()


def test_gate2_fails_when_signature_does_not_match_schema():
    model = ProbabilityModel(np.array([[0.2, 0.8], [0.7, 0.3]]))

    result = _evaluate(model, model, candidate_signature=["feature_a"])

    assert not result.passed
    assert "signature mismatch" in result.summary()


def test_gate2_fails_on_non_finite_predictions():
    candidate = ProbabilityModel(np.array([[0.2, 0.8], [np.nan, np.nan]]))
    production = ProbabilityModel(np.array([[0.3, 0.7], [0.6, 0.4]]))

    result = _evaluate(candidate, production)

    assert not result.passed
    assert "non-finite" in result.summary()


def test_gate2_fails_on_invalid_prediction_range():
    candidate = ProbabilityModel(np.array([[0.2, 0.8], [-0.1, 1.1]]))
    production = ProbabilityModel(np.array([[0.3, 0.7], [0.6, 0.4]]))

    result = _evaluate(candidate, production)

    assert not result.passed
    assert "outside [0, 1]" in result.summary()


def test_gate2_fails_when_prediction_raises():
    candidate = ProbabilityModel(error=RuntimeError("model cannot predict"))
    production = ProbabilityModel(np.array([[0.3, 0.7], [0.6, 0.4]]))

    result = _evaluate(candidate, production)

    assert not result.passed
    assert "prediction failed" in result.summary()
