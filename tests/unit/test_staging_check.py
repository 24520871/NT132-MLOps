"""Gate 2 logic on stub models (no MLflow)."""
import numpy as np
import pandas as pd
import pytest

from mlops.training.staging_check import evaluate_staging

CFG = {
    "min_positive_rate": 0.05,
    "max_positive_rate": 0.70,
    "max_mean_abs_diff_vs_production": 0.25,
    "max_p95_latency_ms": 500,
}


class Stub:
    """predict_proba returns a fixed positive-class probability per row (or a function of the row index)."""

    def __init__(self, proba):
        self.proba = proba

    def predict_proba(self, frame):
        p = np.full(len(frame), self.proba) if np.isscalar(self.proba) else np.asarray(self.proba)[: len(frame)]
        return np.column_stack([1 - p, p])


class Alternating(Stub):
    """30 % of rows above the 0.5 threshold, whatever the frame length."""

    def predict_proba(self, frame):
        p = np.where(np.arange(len(frame)) % 10 < 3, 0.8, 0.2)
        return np.column_stack([1 - p, p])


class Broken:
    def predict_proba(self, frame):
        raise ValueError("boom")


@pytest.fixture
def sample():
    return pd.DataFrame({"a": range(50), "b": range(50)})


def test_accepts_a_sane_model(sample):
    d = evaluate_staging(Alternating(0), sample, CFG)
    assert d.accepted and "p95 latency" in d.summary()


def test_rejects_nan_probabilities(sample):
    d = evaluate_staging(Stub(float("nan")), sample, CFG)
    assert not d.accepted and "NaN" in d.summary()


def test_rejects_out_of_range_probabilities(sample):
    assert not evaluate_staging(Stub(1.5), sample, CFG).accepted


@pytest.mark.parametrize("proba", [0.0, 1.0])  # never / always positive
def test_rejects_implausible_positive_rate(sample, proba):
    d = evaluate_staging(Stub(proba), sample, CFG)
    assert not d.accepted and "positive rate" in d.summary()


def test_rejects_prediction_error(sample):
    d = evaluate_staging(Broken(), sample, CFG)
    assert not d.accepted and "boom" in d.summary()


def test_rejects_signature_mismatch(sample):
    d = evaluate_staging(Alternating(0), sample, CFG, signature_columns=["a", "c"])
    assert not d.accepted and "signature" in d.summary()


def test_accepts_matching_signature_in_any_order(sample):
    assert evaluate_staging(Alternating(0), sample, CFG, signature_columns=["b", "a"]).accepted


def test_shadow_comparison_rejects_big_disagreement(sample):
    production = Stub(0.05)  # staging says 0.2 / 0.8, production ~0.05: mean |dp| ~ 0.3
    d = evaluate_staging(Alternating(0), sample, CFG, production_model=production)
    assert not d.accepted and "vs production" in d.summary()


def test_shadow_comparison_accepts_similar_model(sample):
    d = evaluate_staging(Alternating(0), sample, CFG, production_model=Alternating(0))
    assert d.accepted and "vs production 0.000" in d.summary()


def test_rejects_slow_model(sample):
    d = evaluate_staging(Alternating(0), sample, {**CFG, "max_p95_latency_ms": 0.0})
    assert not d.accepted and "latency" in d.summary()


def test_reports_every_failure(sample):
    d = evaluate_staging(Stub(1.0), sample, {**CFG, "max_p95_latency_ms": 0.0}, production_model=Stub(0.0))
    assert len(d.reasons) == 3
