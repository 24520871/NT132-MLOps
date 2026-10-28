from mlops.training.quality_gate import evaluate_gate

GATE = {"min_auc": 0.80, "min_recall": 0.60, "max_auc_drop_vs_production": 0.0}


def test_accepts_when_thresholds_met_and_no_production():
    d = evaluate_gate({"auc": 0.84, "recall": 0.70}, GATE)
    assert d.accepted and "no production" in d.summary()


def test_rejects_low_auc():
    d = evaluate_gate({"auc": 0.75, "recall": 0.90}, GATE)
    assert not d.accepted and "auc" in d.summary()


def test_rejects_low_recall():
    d = evaluate_gate({"auc": 0.90, "recall": 0.50}, GATE)
    assert not d.accepted and "recall" in d.summary()


def test_reports_every_failure():
    d = evaluate_gate({"auc": 0.5, "recall": 0.1}, GATE)
    assert len(d.reasons) == 2


def test_rejects_worse_than_production():
    d = evaluate_gate({"auc": 0.82, "recall": 0.70}, GATE, production={"auc": 0.85, "recall": 0.7})
    assert not d.accepted and "production" in d.summary()


def test_accepts_equal_to_production():
    assert evaluate_gate({"auc": 0.85, "recall": 0.70}, GATE, production={"auc": 0.85}).accepted


def test_tolerance_allows_small_drop():
    gate = {**GATE, "max_auc_drop_vs_production": 0.01}
    assert evaluate_gate({"auc": 0.845, "recall": 0.7}, gate, production={"auc": 0.85}).accepted
