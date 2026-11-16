"""Prediction log: what the drift monitor reads."""
import json

from mlops.serving.prediction_log import PredictionLog, read_prediction_log


def test_record_and_read_roundtrip(tmp_path):
    path = tmp_path / "logs" / "predictions.jsonl"  # parent dir is created on demand
    plog = PredictionLog(path)
    plog.record({"tenure": 5, "Contract": "Month-to-month"}, 0.8, 1, "churn", "3")
    plog.record({"tenure": 40, "Contract": "Two year"}, 0.1, 0, "churn", "3")
    df = read_prediction_log(path)
    assert len(df) == 2
    assert {"ts", "model_version", "probability", "prediction", "tenure", "Contract"} <= set(df.columns)
    assert df["tenure"].tolist() == [5, 40]
    assert df["model_version"].tolist() == ["3", "3"]


def test_each_line_is_valid_json(tmp_path):
    plog = PredictionLog(tmp_path / "p.jsonl")
    plog.record({"tenure": 1}, 0.5, 1, "churn", "1")
    line = (tmp_path / "p.jsonl").read_text().splitlines()[0]
    assert json.loads(line)["features"] == {"tenure": 1}


def test_write_failure_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    PredictionLog(blocker / "sub" / "p.jsonl").record({"a": 1}, 0.5, 1, "churn", "1")  # parent is a file


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("PREDICTION_LOG_PATH", raising=False)
    assert PredictionLog.from_env() is None
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "p.jsonl"))
    assert PredictionLog.from_env().path == tmp_path / "p.jsonl"
