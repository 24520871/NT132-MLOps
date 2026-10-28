import pytest

from mlops.training.evaluate import evaluate_classifier


def test_perfect_predictions():
    m, cm = evaluate_classifier([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m == {"auc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert cm == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_known_confusion_matrix_and_threshold():
    y = [0, 0, 1, 1]
    p = [0.6, 0.2, 0.4, 0.9]  # threshold 0.5 -> pred 1,0,0,1
    m, cm = evaluate_classifier(y, p, threshold=0.5)
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    m2, _ = evaluate_classifier(y, p, threshold=0.3)  # more positives -> recall up
    assert m2["recall"] == 1.0


def test_no_positive_predictions_does_not_crash():
    m, _ = evaluate_classifier([0, 1, 0, 1], [0.1, 0.2, 0.1, 0.3])
    assert m["precision"] == 0.0 and m["recall"] == 0.0
