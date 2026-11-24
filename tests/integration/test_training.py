"""Train on a small fake frame in an isolated MLflow store, then run both gates (candidate -> staging -> production)."""
import mlflow
import pytest
from mlflow.tracking import MlflowClient

from mlops.registry.promote import NoCandidateError, NoStagingError, promote_staging, stage_candidate
from mlops.training.runner import run_training
from tests.conftest import LENIENT_STAGING, make_raw_frame
from usecases.churn import data
from usecases.churn.model import build_models

LENIENT = {"min_auc": 0.0, "min_recall": 0.0, "max_auc_drop_vs_production": 0.0}
IMPOSSIBLE = {"min_auc": 0.999, "min_recall": 0.999, "max_auc_drop_vs_production": 0.0}


@pytest.fixture
def trained(mlflow_store, churn_cfg):
    df = data.clean(make_raw_frame(n=600))
    X_tr, X_te, y_tr, y_te = data.split_data(df, churn_cfg)
    return run_training(churn_cfg, build_models(churn_cfg, y_tr), X_tr, X_te, y_tr, y_te, "abc123")


def _held_out(churn_cfg, seed=0):
    """Feature rows standing in for the held-out sample Gate 2 receives."""
    return data.split_data(data.clean(make_raw_frame(n=600, seed=seed)), churn_cfg)[1]


def _alias_version(name, alias):
    try:
        return str(MlflowClient().get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def _train(churn_cfg, seed):
    """Train on a fresh fake frame; the best model becomes the new `candidate`."""
    df = data.clean(make_raw_frame(n=600, seed=seed))
    X_tr, X_te, y_tr, y_te = data.split_data(df, churn_cfg)
    return run_training(churn_cfg, build_models(churn_cfg, y_tr), X_tr, X_te, y_tr, y_te)


def _promote_to_production(churn_cfg, seed):
    """Train, then pass both gates (lenient). Returns the new production version."""
    run = _train(churn_cfg, seed)
    name = churn_cfg["model_name"]
    stage_candidate(name, {**LENIENT, "max_auc_drop_vs_production": 1.0})
    res = promote_staging(name, LENIENT_STAGING, _held_out(churn_cfg), churn_cfg["threshold"])
    assert res.decision.accepted
    return run.version


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


def test_accepted_candidate_moves_to_staging_not_production(trained, churn_cfg):
    name = churn_cfg["model_name"]
    res = stage_candidate(name, LENIENT)
    assert res.decision.accepted
    assert _alias_version(name, "staging") == trained.version
    assert _alias_version(name, "production") is None  # Gate 2 has not run yet
    assert _alias_version(name, "candidate") is None  # candidate alias consumed
    assert MlflowClient().get_model_version(name, trained.version).tags["gate"] == "accepted"


def test_staging_version_becomes_production_after_gate2(trained, churn_cfg):
    name = churn_cfg["model_name"]
    stage_candidate(name, LENIENT)
    res = promote_staging(name, LENIENT_STAGING, _held_out(churn_cfg), churn_cfg["threshold"])
    assert res.decision.accepted and res.previous_production is None
    assert _alias_version(name, "production") == trained.version
    assert _alias_version(name, "staging") is None
    assert MlflowClient().get_model_version(name, trained.version).tags["staging_gate"] == "accepted"


def test_rejected_candidate_never_reaches_staging(trained, churn_cfg):
    name = churn_cfg["model_name"]
    res = stage_candidate(name, IMPOSSIBLE)
    assert not res.decision.accepted
    assert _alias_version(name, "staging") is None and _alias_version(name, "production") is None
    mv = MlflowClient().get_model_version(name, trained.version)
    assert mv.tags["gate"] == "rejected" and "auc" in mv.tags["gate_reasons"]


def test_gate2_rejection_keeps_production_and_clears_staging(mlflow_store, churn_cfg):
    name = churn_cfg["model_name"]
    first = _promote_to_production(churn_cfg, seed=0)
    second = _train(churn_cfg, seed=1)
    stage_candidate(name, {**LENIENT, "max_auc_drop_vs_production": 1.0})
    strict = {**LENIENT_STAGING, "max_positive_rate": 0.0}  # nothing may be predicted positive -> must fail
    res = promote_staging(name, strict, _held_out(churn_cfg), churn_cfg["threshold"])
    assert not res.decision.accepted
    assert _alias_version(name, "production") == first  # untouched
    assert _alias_version(name, "staging") is None
    mv = MlflowClient().get_model_version(name, second.version)
    assert mv.tags["staging_gate"] == "rejected" and "positive rate" in mv.tags["staging_gate_reasons"]


def test_gate1_rejection_keeps_existing_production(mlflow_store, churn_cfg):
    name = churn_cfg["model_name"]
    first = _promote_to_production(churn_cfg, seed=0)
    _train(churn_cfg, seed=1)
    res = stage_candidate(name, IMPOSSIBLE)
    assert not res.decision.accepted and res.previous_production == first
    assert _alias_version(name, "production") == first
    assert _alias_version(name, "staging") is None


def test_second_promotion_records_previous_production(mlflow_store, churn_cfg):
    name = churn_cfg["model_name"]
    first = _promote_to_production(churn_cfg, seed=0)
    second = _promote_to_production(churn_cfg, seed=2)
    assert _alias_version(name, "production") == second
    assert _alias_version(name, "previous_production") == first


def test_stage_without_candidate_errors(mlflow_store, churn_cfg):
    with pytest.raises(NoCandidateError, match="Run training first"):
        stage_candidate(churn_cfg["model_name"], LENIENT)


def test_gate2_without_staging_version_errors(trained, churn_cfg):
    with pytest.raises(NoStagingError, match="Run Gate 1 first"):
        promote_staging(churn_cfg["model_name"], LENIENT_STAGING, _held_out(churn_cfg), churn_cfg["threshold"])
