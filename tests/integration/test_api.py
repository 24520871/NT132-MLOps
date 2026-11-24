"""API contract tests against an isolated MLflow store (train + promote a tiny model)."""
import mlflow
import pytest
from fastapi.testclient import TestClient

from mlops.registry.promote import promote_staging, stage_candidate
from mlops.serving.app import create_app
from mlops.serving.router import ModelRouter
from mlops.training.runner import run_training
from tests.conftest import LENIENT_GATE, LENIENT_STAGING, make_raw_frame, small_cfg
from usecases.churn import data
from usecases.churn.model import build_models

SAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5, "TotalCharges": 447.5,
}  # fmt: skip


def _train_and_promote(cfg, seed=0):
    df = data.clean(make_raw_frame(n=600, seed=seed))
    X_tr, X_te, y_tr, y_te = data.split_data(df, cfg)
    run_training(cfg, build_models(cfg, y_tr), X_tr, X_te, y_tr, y_te)
    stage_candidate(cfg["model_name"], LENIENT_GATE)
    promote_staging(cfg["model_name"], LENIENT_STAGING, X_te, cfg["threshold"])


def _use_store(monkeypatch, tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", (tmp_path / "artifacts").as_posix())
    mlflow.set_tracking_uri(uri)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """One store with a promoted v1, shared by the tests in this module."""
    mp = pytest.MonkeyPatch()
    _use_store(mp, tmp_path_factory.mktemp("served"))
    cfg = small_cfg()
    _train_and_promote(cfg)
    yield TestClient(create_app(ModelRouter(cfg["model_name"]), cfg)), cfg
    mp.undo()


@pytest.fixture
def empty_client(mlflow_store, churn_cfg):
    return TestClient(create_app(ModelRouter(churn_cfg["model_name"]), churn_cfg))


def test_health_is_up_without_a_model(empty_client):
    r = empty_client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ready_503_without_production_model(empty_client):
    r = empty_client.get("/ready")
    assert r.status_code == 503
    assert "Train and promote" in r.json()["detail"]


def test_predict_503_without_production_model(empty_client):
    r = empty_client.post("/predict/churn", json=SAMPLE)
    assert r.status_code == 503
    assert "promote" in r.json()["detail"]


def test_ready_200_with_model(served):
    client, _ = served
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_version"] == "1" and r.json()["status"] == "ready"


def test_valid_prediction(served):
    client, cfg = served
    r = client.post("/predict/churn", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] == int(body["probability"] >= cfg["threshold"])
    assert body["label"] == ("churn" if body["prediction"] else "no_churn")
    assert body["model_version"] == "1" and body["model_name"] == "churn"


def test_prediction_is_deterministic(served):
    client, _ = served
    a = client.post("/predict/churn", json=SAMPLE).json()
    b = client.post("/predict/churn", json=SAMPLE).json()
    assert a == b


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("Contract"),                          # missing field
        lambda p: p.update(Contract="Three year"),            # bad category
        lambda p: p.update(tenure=-1),                        # below range
        lambda p: p.update(tenure=500),                       # above range
        lambda p: p.update(MonthlyCharges="abc"),             # wrong type
        lambda p: p.update(SeniorCitizen=2),                  # not 0/1
        lambda p: p.update(unexpected_field=1),               # extra field
    ],
)
def test_invalid_requests_return_422(served, mutate):
    client, _ = served
    payload = dict(SAMPLE)
    mutate(payload)
    assert client.post("/predict/churn", json=payload).status_code == 422


def test_empty_body_returns_422(served):
    client, _ = served
    assert client.post("/predict/churn", json={}).status_code == 422


def test_docs_available(served):
    client, _ = served
    assert client.get("/docs").status_code == 200
    assert "/predict/churn" in client.get("/openapi.json").json()["paths"]


def test_admin_reload_picks_up_new_production_version(served):
    client, cfg = served
    _train_and_promote(cfg, seed=3)  # promotes v2; the router still trusts its cached alias lookup
    r = client.post("/admin/reload")
    assert r.status_code == 200 and r.json()["model_version"] == "2"
    assert client.post("/predict/churn", json=SAMPLE).json()["model_version"] == "2"
    assert client.get("/ready").json()["model_version"] == "2"


def test_admin_reload_requires_token_when_configured(served, monkeypatch):
    client, _ = served
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.post("/admin/reload").status_code == 401
    assert client.post("/admin/reload", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.post("/admin/reload", headers={"X-Admin-Token": "s3cret"}).status_code == 200


def test_admin_reload_503_without_production_model(empty_client):
    assert empty_client.post("/admin/reload").status_code == 503


def test_router_polls_alias_and_caches_between_polls(mlflow_store, churn_cfg):
    name = churn_cfg["model_name"]
    _train_and_promote(churn_cfg)  # v1 -> production
    polling = ModelRouter(name, poll_seconds=0)  # checks on every call
    cached = ModelRouter(name, poll_seconds=3600)  # trusts the first lookup
    assert polling.get()[1] == "1" and cached.get()[1] == "1"
    _train_and_promote(churn_cfg, seed=5)  # v2 -> production
    assert polling.get()[1] == "2"  # alias poll noticed the new version
    assert cached.get()[1] == "1"  # still within the poll window
    assert cached.reload()[1] == "2"  # reload forces a check
