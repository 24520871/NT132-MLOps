"""/metrics and the prediction log on the real app, against an isolated MLflow store."""
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from mlops.serving.app import create_app
from mlops.serving.prediction_log import PredictionLog, read_prediction_log
from mlops.serving.router import ModelRouter
from tests.conftest import small_cfg
from tests.integration.test_api import SAMPLE, _train_and_promote, _use_store


def value(client, name, **labels) -> float | None:
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for s in family.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return None


@pytest.fixture
def monitored(tmp_path, monkeypatch):
    _use_store(monkeypatch, tmp_path)
    cfg = small_cfg()
    _train_and_promote(cfg)
    log_path = tmp_path / "predictions.jsonl"
    client = TestClient(create_app(ModelRouter(cfg["model_name"]), cfg, PredictionLog(log_path)))
    return client, log_path


def test_metrics_endpoint_exposes_text_format(monitored):
    client, _ = monitored
    r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")


def test_prediction_updates_metrics(monitored):
    client, _ = monitored
    assert client.post("/predict/churn", json=SAMPLE).status_code == 200
    assert value(client, "api_requests_total", endpoint="/predict/churn", status="200") == 1
    assert value(client, "api_request_duration_seconds_count", endpoint="/predict/churn") == 1
    assert value(client, "churn_probability_count") == 1
    label = client.post("/predict/churn", json=SAMPLE).json()["label"]
    assert value(client, "churn_predictions_total", label=label) == 2
    assert value(client, "model_version_info", model="churn", version="1") == 1


def test_client_errors_are_counted_but_not_logged(monitored):
    client, log_path = monitored
    assert client.post("/predict/churn", json={}).status_code == 422
    assert value(client, "api_requests_total", endpoint="/predict/churn", status="422") == 1
    assert value(client, "churn_probability_count") == 0
    assert not log_path.exists()


def test_unknown_paths_share_one_label(monitored):
    client, _ = monitored
    client.get("/nope-1"), client.get("/nope-2")
    assert value(client, "api_requests_total", endpoint="unmatched", status="404") == 2


def test_metrics_endpoint_does_not_count_itself(monitored):
    client, _ = monitored
    client.get("/metrics")
    assert value(client, "api_requests_total", endpoint="/metrics") is None


def test_503_without_model_is_counted(tmp_path, monkeypatch):
    _use_store(monkeypatch, tmp_path)
    cfg = small_cfg()
    client = TestClient(create_app(ModelRouter(cfg["model_name"]), cfg))
    assert client.post("/predict/churn", json=SAMPLE).status_code == 503
    assert value(client, "api_requests_total", endpoint="/predict/churn", status="503") == 1


def test_prediction_is_logged_with_features(monitored):
    client, log_path = monitored
    body = client.post("/predict/churn", json=SAMPLE).json()
    df = read_prediction_log(log_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["probability"] == pytest.approx(body["probability"])
    assert row["prediction"] == body["prediction"] and row["model_version"] == "1"
    assert row["tenure"] == SAMPLE["tenure"] and row["Contract"] == SAMPLE["Contract"]
