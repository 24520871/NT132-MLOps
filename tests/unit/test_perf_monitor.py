import json

import pandas as pd
import pytest

from mlops.monitoring.perf_monitor import evaluate_performance


def test_calculates_recall_and_auc_with_valid_ground_truth():
    records = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "prediction": [0, 0, 1, 0],
            "probability": [0.1, 0.2, 0.8, 0.7],
        }
    )

    result = evaluate_performance(records)

    assert result.available
    assert result.sample_count == 4
    assert result.recall == pytest.approx(0.5)
    assert result.auc == pytest.approx(1.0)
    assert result.reason is None


def test_reads_prediction_log_without_inventing_missing_labels(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps(
            {
                "ts": "2026-11-17T00:00:00Z",
                "model_name": "churn",
                "model_version": "1",
                "probability": 0.8,
                "prediction": 1,
                "features": {"tenure": 5},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_performance(path)

    assert not result.available
    assert result.recall is None and result.auc is None
    assert "ground-truth labels are unavailable" in result.reason


def test_reads_valid_ground_truth_embedded_in_prediction_log(tmp_path):
    path = tmp_path / "labeled-predictions.jsonl"
    rows = [
        {"ts": "2026-11-17T00:00:00Z", "prediction": 0, "probability": 0.1, "y_true": 0},
        {"ts": "2026-11-17T00:00:01Z", "prediction": 0, "probability": 0.2, "y_true": 0},
        {"ts": "2026-11-17T00:00:02Z", "prediction": 1, "probability": 0.8, "y_true": 1},
        {"ts": "2026-11-17T00:00:03Z", "prediction": 0, "probability": 0.7, "y_true": 1},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = evaluate_performance(path)

    assert result.recall == pytest.approx(0.5)
    assert result.auc == pytest.approx(1.0)


def test_empty_prediction_log_is_unavailable():
    result = evaluate_performance(pd.DataFrame())

    assert not result.available
    assert result.reason == "prediction log is empty"


def test_missing_label_values_are_rejected():
    records = pd.DataFrame(
        {
            "y_true": [0, None],
            "prediction": [0, 1],
            "probability": [0.1, 0.9],
        }
    )

    with pytest.raises(ValueError, match="labels contain missing"):
        evaluate_performance(records)


def test_auc_is_unavailable_when_ground_truth_has_only_one_class():
    records = pd.DataFrame(
        {
            "y_true": [0, 0, 0],
            "prediction": [0, 1, 0],
            "probability": [0.1, 0.9, 0.2],
        }
    )

    result = evaluate_performance(records)

    assert result.recall is None
    assert result.auc is None
    assert not result.available
    assert "both classes" in result.reason


def test_recall_can_be_reported_but_auc_unavailable_without_negatives():
    records = pd.DataFrame(
        {
            "ground_truth": ["Yes", "Yes", "Yes"],
            "prediction": [1, 0, 1],
            "probability": [0.9, 0.4, 0.8],
        }
    )

    result = evaluate_performance(records)

    assert result.recall == pytest.approx(2 / 3)
    assert result.auc is None
    assert "both classes" in result.reason


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("prediction", [0, 2], "predictions must contain only"),
        ("probability", [0.1, 1.2], "probabilities must be finite"),
    ],
)
def test_rejects_invalid_predictions_and_probabilities(column, values, message):
    records = pd.DataFrame(
        {
            "y_true": [0, 1],
            "prediction": [0, 1],
            "probability": [0.1, 0.9],
        }
    )
    records[column] = values

    with pytest.raises(ValueError, match=message):
        evaluate_performance(records)
