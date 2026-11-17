import numpy as np
import pandas as pd
import pytest
from prometheus_client.parser import text_string_to_metric_families

from mlops.monitoring.drift import (
    detect_drift,
    write_html_report,
    write_prometheus_metrics,
)
from mlops.monitoring.perf_monitor import PerformanceResult
from scripts.replay_drift import generate_replay_frames, run_replay


def _families(path):
    return {
        sample.name: sample.value
        for family in text_string_to_metric_families(path.read_text(encoding="utf-8"))
        for sample in family.samples
    }


def test_identical_feature_distributions_have_no_drift():
    values = np.linspace(0, 1, 100)
    reference = pd.DataFrame({"value": values, "category": ["a", "b"] * 50})

    result = detect_drift(reference, reference.copy())

    assert result.data_drift_share == 0
    assert not result.data_drift_detected
    assert result.drifted_features == ()


def test_detects_shifted_numeric_and_categorical_features():
    reference = pd.DataFrame(
        {
            "numeric": np.linspace(0, 1, 500),
            "category": ["a"] * 250 + ["b"] * 250,
        }
    )
    current = pd.DataFrame(
        {
            "numeric": np.linspace(4, 5, 500),
            "category": ["a"] * 450 + ["b"] * 50,
        }
    )

    result = detect_drift(reference, current)

    assert result.data_drift_detected
    assert result.data_drift_share == 1
    assert set(result.drifted_features) == {"numeric", "category"}


@pytest.mark.parametrize(
    ("reference", "current", "message"),
    [
        (pd.DataFrame(), pd.DataFrame({"x": [1, 2]}), "must not be empty"),
        (pd.DataFrame({"x": [1, 2]}), pd.DataFrame(), "must not be empty"),
        (
            pd.DataFrame({"x": [1, 2], "y": [2, 3]}),
            pd.DataFrame({"x": [1, 2]}),
            "incompatible feature columns",
        ),
    ],
)
def test_rejects_empty_or_incompatible_data(reference, current, message):
    with pytest.raises(ValueError, match=message):
        detect_drift(reference, current)


def test_target_and_metadata_are_not_drift_features():
    features = np.linspace(0, 1, 100)
    reference = pd.DataFrame(
        {
            "feature": features,
            "Churn": np.zeros(100, dtype=int),
            "ts": ["old"] * 100,
            "model_version": ["1"] * 100,
            "prediction": np.zeros(100, dtype=int),
            "probability": np.full(100, 0.1),
        }
    )
    current = pd.DataFrame(
        {
            "feature": features,
            "Churn": np.ones(100, dtype=int),
            "ts": ["new"] * 100,
            "model_version": ["99"] * 100,
            "prediction": np.ones(100, dtype=int),
            "probability": np.full(100, 0.99),
        }
    )

    result = detect_drift(reference, current)

    assert [feature.name for feature in result.feature_results] == ["feature"]
    assert not result.data_drift_detected


def test_rejects_incompatible_feature_types():
    reference = pd.DataFrame({"feature": [1.0, 2.0]})
    current = pd.DataFrame({"feature": ["one", "two"]})

    with pytest.raises(ValueError, match="incompatible numeric/categorical"):
        detect_drift(reference, current)


def test_writes_prometheus_drift_and_labeled_performance_metrics(tmp_path):
    result = detect_drift(
        pd.DataFrame({"x": np.arange(100)}),
        pd.DataFrame({"x": np.arange(100) + 100}),
    )
    performance = PerformanceResult(recall=0.75, auc=0.9, sample_count=20)
    path = write_prometheus_metrics(
        result,
        tmp_path / "metrics" / "drift.prom",
        simulated=True,
        performance=performance,
    )

    metrics = _families(path)
    assert metrics["data_drift_share"] == 1
    assert metrics["data_drift_detected"] == 1
    assert metrics["data_drift_simulated"] == 1
    assert metrics["model_recall"] == 0.75
    assert metrics["model_auc"] == 0.9


def test_unavailable_performance_does_not_export_fake_metrics(tmp_path):
    result = detect_drift(
        pd.DataFrame({"x": np.arange(20)}),
        pd.DataFrame({"x": np.arange(20)}),
    )
    path = tmp_path / "drift.prom"
    write_prometheus_metrics(
        result,
        path,
        performance=PerformanceResult(recall=0.5, auc=0.6, sample_count=10),
    )
    write_prometheus_metrics(
        result,
        path,
        performance=PerformanceResult(None, None, 0, "no labels"),
    )

    metrics = _families(path)
    assert "model_recall" not in metrics
    assert "model_auc" not in metrics


def test_writes_html_report_and_marks_simulation(tmp_path):
    result = detect_drift(
        pd.DataFrame({"x": np.arange(20)}),
        pd.DataFrame({"x": np.arange(20) + 100}),
    )

    path = write_html_report(result, tmp_path / "report.html", simulated=True)

    assert "SIMULATED replay data" in path.read_text(encoding="utf-8")
    assert "DRIFT DETECTED" in path.read_text(encoding="utf-8")


def test_synthetic_replay_is_reproducible_and_outputs_artifacts(tmp_path):
    a = generate_replay_frames(rows=200, seed=7)
    b = generate_replay_frames(rows=200, seed=7)
    for left, right in zip(a, b):
        pd.testing.assert_frame_equal(left, right)

    summary = run_replay(
        rows=200,
        seed=7,
        report_dir=tmp_path / "reports",
        metrics_path=tmp_path / "metrics" / "drift.prom",
        prediction_log_path=tmp_path / "missing-predictions.jsonl",
    )

    assert summary["simulation"] == "simulated"
    assert summary["shifted"]["data_drift_detected"] == 1
    assert summary["performance"]["recall"] is None
    assert (tmp_path / "reports" / "replay-normal.html").exists()
    assert (tmp_path / "reports" / "replay-shifted.html").exists()
    assert (tmp_path / "metrics" / "drift.prom").exists()
