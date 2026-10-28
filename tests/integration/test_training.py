"""Train on a small fake frame in an isolated MLflow store, then gate/promote."""
import mlflow
import pytest
from mlflow.tracking import MlflowClient

from mlops.registry.promote import NoCandidateError, promote_candidate
from mlops.training.runner import run_training
from tests.conftest import make_raw_frame
from usecases.churn import data
from usecases.churn.model import build_models

LENIENT = {"min_auc": 0.0, "min_recall": 0.0, "max_auc_drop_vs_production": 0.0}
IMPOSSIBLE = {"min_auc": 0.999, "min_recall": 0.999, "max_auc_drop_vs_production": 0.0}


@pytest.fixture
def trained(mlflow_store, churn_cfg):
    df = data.clean(make_raw_frame(n=600))
    X_tr, X_te, y_tr, y_te = data.split_data(df, churn_cfg)
    return run_training(churn_cfg, build_models(churn_cfg, y_tr), X_tr, X_te, y_tr, y_te, "abc123")


def test_logs_one_run_per_model_with_metrics(trained, churn_cfg):
    runs = mlflow.search_runs(experiment_names=[churn_cfg["experiment"]])
    assert len(runs) == 2
    for col in ("metrics.auc", "metrics.precision", "metrics.recall", "metrics.f1"):
        assert runs[col].notna().all()
    assert (runs["params.data_sha256"] == "abc123").all()


def test_best_model_registered_as_candidate_only(trained, churn_cfg):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(churn_cfg["model_name"], "candidate")
    assert str(mv.version) == trained.version
    assert trained.metrics["auc"] == max(m["auc"] for m in trained.all_metrics.values())
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(churn_cfg["model_name"], "production")


def test_accepted_candidate_becomes_production(trained, churn_cfg):
    res = promote_candidate(churn_cfg["model_name"], LENIENT)
    assert res.decision.accepted
    client = MlflowClient()
    assert str(client.get_model_version_by_alias(churn_cfg["model_name"], "production").version) == trained.version
    with pytest.raises(mlflow.exceptions.MlflowException):  # candidate alias consumed
        client.get_model_version_by_alias(churn_cfg["model_name"], "candidate")


def test_rejected_candidate_is_not_promoted(trained, churn_cfg):
    res = promote_candidate(churn_cfg["model_name"], IMPOSSIBLE)
    assert not res.decision.accepted
    client = MlflowClient()
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(churn_cfg["model_name"], "production")
    mv = client.get_model_version(churn_cfg["model_name"], trained.version)
    assert mv.tags["gate"] == "rejected" and "auc" in mv.tags["gate_reasons"]


def test_rejected_candidate_keeps_existing_production(trained, churn_cfg):
    name = churn_cfg["model_name"]
    promote_candidate(name, LENIENT)  # v1 -> production
    df = data.clean(make_raw_frame(n=600, seed=1))
    X_tr, X_te, y_tr, y_te = data.split_data(df, churn_cfg)
    second = run_training(churn_cfg, build_models(churn_cfg, y_tr), X_tr, X_te, y_tr, y_te)
    res = promote_candidate(name, IMPOSSIBLE)
    assert not res.decision.accepted and res.previous_production == trained.version
    assert str(MlflowClient().get_model_version_by_alias(name, "production").version) == trained.version
    assert second.version != trained.version


def test_second_promotion_records_previous_production(trained, churn_cfg):
    name = churn_cfg["model_name"]
    promote_candidate(name, LENIENT)
    df = data.clean(make_raw_frame(n=600, seed=2))
    X_tr, X_te, y_tr, y_te = data.split_data(df, churn_cfg)
    second = run_training(churn_cfg, build_models(churn_cfg, y_tr), X_tr, X_te, y_tr, y_te)
    lenient_any = {**LENIENT, "max_auc_drop_vs_production": 1.0}
    assert promote_candidate(name, lenient_any).decision.accepted
    client = MlflowClient()
    assert str(client.get_model_version_by_alias(name, "production").version) == second.version
    assert str(client.get_model_version_by_alias(name, "previous_production").version) == trained.version


def test_promote_without_candidate_errors(mlflow_store, churn_cfg):
    with pytest.raises(NoCandidateError, match="Run training first"):
        promote_candidate(churn_cfg["model_name"], LENIENT)
