"""Train fixture models and exercise the staged MLflow promotion lifecycle."""
import mlflow
import pytest
from mlflow.tracking import MlflowClient

from mlops.registry.promote import (
    GATE1_TAG,
    GATE2_TAG,
    PRODUCTION,
    STAGING,
    NoCandidateError,
    gate1_candidate,
    gate2_staging,
    promote_staging,
)
from mlops.training.runner import run_training
from tests.conftest import make_raw_frame
from usecases.churn import data
from usecases.churn.model import build_models

LENIENT = {"min_auc": 0.0, "min_recall": 0.0, "max_auc_drop_vs_production": 1.0}
IMPOSSIBLE = {"min_auc": 0.999, "min_recall": 0.999, "max_auc_drop_vs_production": 0.0}


@pytest.fixture
def trained(mlflow_store, churn_cfg):
    df = data.clean(make_raw_frame(n=600))
    X_tr, X_te, y_tr, y_te = data.split_data(df, churn_cfg)
    result = run_training(churn_cfg, build_models(churn_cfg, y_tr), X_tr, X_te, y_tr, y_te, "abc123")
    return result


def _baseline(name, version):
    """Seed a production baseline for integration tests; Gate 2 requires one."""
    MlflowClient().set_registered_model_alias(name, PRODUCTION, version)


def _run_accepted_flow(name, cfg):
    gate1 = gate1_candidate(name, cfg["gate"])
    assert gate1.decision.accepted
    gate2 = gate2_staging(cfg)
    assert gate2.decision.accepted, gate2.decision.summary()
    promoted = promote_staging(name)
    assert promoted.decision.accepted
    return promoted


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
        client.get_model_version_by_alias(churn_cfg["model_name"], PRODUCTION)


def test_gate1_acceptance_moves_candidate_to_staging_only(trained, churn_cfg):
    name = churn_cfg["model_name"]
    result = gate1_candidate(name, LENIENT)
    assert result.decision.accepted
    client = MlflowClient()
    staged = client.get_model_version_by_alias(name, STAGING)
    assert str(staged.version) == trained.version
    assert client.get_model_version(name, trained.version).tags[GATE1_TAG] == "accepted"
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(name, PRODUCTION)


def test_gate1_rejection_leaves_production_unchanged(trained, churn_cfg):
    name = churn_cfg["model_name"]
    _baseline(name, trained.version)
    result = gate1_candidate(name, IMPOSSIBLE)
    assert not result.decision.accepted
    client = MlflowClient()
    assert str(client.get_model_version_by_alias(name, PRODUCTION).version) == trained.version
    assert client.get_model_version(name, trained.version).tags["gate"] == "rejected"


def test_gate2_without_production_baseline_does_not_promote(trained, churn_cfg):
    name = churn_cfg["model_name"]
    assert gate1_candidate(name, LENIENT).decision.accepted
    result = gate2_staging(churn_cfg)
    assert not result.decision.accepted
    assert "production baseline is required" in result.decision.summary()
    assert MlflowClient().get_model_version(name, trained.version).tags[GATE2_TAG] == "rejected"


def test_complete_flow_promotes_and_records_previous_production(trained, churn_cfg):
    name = churn_cfg["model_name"]
    _baseline(name, trained.version)
    result = _run_accepted_flow(name, churn_cfg)
    assert result.version == trained.version
    assert str(MlflowClient().get_model_version_by_alias(name, PRODUCTION).version) == trained.version


def test_rejected_gate1_preserves_existing_production(trained, churn_cfg):
    name = churn_cfg["model_name"]
    _baseline(name, trained.version)
    df = data.clean(make_raw_frame(n=600, seed=1))
    X_tr, X_te, y_tr, y_te = data.split_data(df, churn_cfg)
    second = run_training(churn_cfg, build_models(churn_cfg, y_tr), X_tr, X_te, y_tr, y_te)
    result = gate1_candidate(name, IMPOSSIBLE)
    assert not result.decision.accepted
    assert str(MlflowClient().get_model_version_by_alias(name, PRODUCTION).version) == trained.version
    assert second.version != trained.version


def test_promote_without_candidate_errors(mlflow_store, churn_cfg):
    with pytest.raises(NoCandidateError, match="Train first"):
        gate1_candidate(churn_cfg["model_name"], churn_cfg["gate"])
