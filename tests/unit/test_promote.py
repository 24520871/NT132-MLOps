from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from mlops.registry import promote
from mlops.training.quality_gate import GateDecision
from usecases.churn.schema import FEATURE_COLUMNS

GATE = {"min_auc": 0.8, "min_recall": 0.6, "max_auc_drop_vs_production": 0.0}
METRICS = {"auc": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75}


class ProbabilityModel:
    def predict_proba(self, features):
        return np.tile(np.array([[0.3, 0.7]]), (len(features), 1))


def _version(version, run_id, tags=None):
    return SimpleNamespace(version=str(version), run_id=run_id, tags=tags or {})


def _client_with_metrics(metrics_by_run):
    client = Mock()
    client.get_run.side_effect = lambda run_id: SimpleNamespace(
        data=SimpleNamespace(metrics=metrics_by_run[run_id])
    )
    return client


def _patch_aliases(monkeypatch, aliases):
    monkeypatch.setattr(
        promote,
        "_version_by_alias",
        lambda _client, _name, alias: aliases.get(alias),
    )


def test_gate1_passes_candidate_into_staging_without_changing_production(monkeypatch):
    client = _client_with_metrics({"candidate-run": METRICS, "production-run": METRICS})
    aliases = {
        promote.CANDIDATE: _version("2", "candidate-run"),
        promote.PRODUCTION: _version("1", "production-run"),
        promote.PREVIOUS: _version("0", "old-run"),
    }
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)

    result = promote.gate1_candidate("churn", GATE)

    assert result.decision.accepted
    client.set_registered_model_alias.assert_called_once_with("churn", "staging", "2")
    client.delete_registered_model_alias.assert_called_once_with("churn", "candidate")
    assert all(call.args[1] != "previous_production" for call in client.set_registered_model_alias.call_args_list)


def test_gate1_rejects_candidate_worse_than_production_without_alias_changes(monkeypatch):
    candidate_metrics = {**METRICS, "auc": 0.89}
    production_metrics = {**METRICS, "auc": 0.90}
    client = _client_with_metrics(
        {"candidate-run": candidate_metrics, "production-run": production_metrics}
    )
    aliases = {
        promote.CANDIDATE: _version("2", "candidate-run"),
        promote.PRODUCTION: _version("1", "production-run"),
    }
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)

    result = promote.gate1_candidate("churn", GATE)

    assert not result.decision.accepted
    assert "worse than production" in result.decision.summary()
    client.set_registered_model_alias.assert_not_called()
    client.delete_registered_model_alias.assert_not_called()


def test_gate1_rejects_missing_metric(monkeypatch):
    client = _client_with_metrics({"candidate-run": {"auc": 0.9}})
    aliases = {promote.CANDIDATE: _version("2", "candidate-run")}
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)

    result = promote.gate1_candidate("churn", GATE)

    assert not result.decision.accepted
    assert "recall" in result.decision.summary()
    client.set_registered_model_alias.assert_not_called()


def test_gate2_requires_production_baseline(monkeypatch):
    client = Mock()
    aliases = {promote.STAGING: _version("2", "staging-run")}
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)

    result = promote.gate2_staging({"name": "churn", "model_name": "churn"})

    assert not result.decision.accepted
    assert "production baseline is required" in result.decision.summary()
    client.get_run.assert_not_called()
    assert client.set_model_version_tag.call_count == 2


def test_gate2_does_not_gate_on_logged_metrics(monkeypatch):
    client = Mock()
    aliases = {
        promote.STAGING: _version("2", "staging-run"),
        promote.PRODUCTION: _version("1", "production-run"),
    }
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)
    features = pd.DataFrame(np.zeros((4, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    monkeypatch.setattr(promote, "_load_sklearn_model", lambda _name, _version: ProbabilityModel())
    monkeypatch.setattr(promote, "_signature_fields", lambda _name, _version: FEATURE_COLUMNS)
    monkeypatch.setattr(promote.churn_data, "load_data", lambda _cfg: pd.DataFrame())
    monkeypatch.setattr(
        promote.churn_data, "split_data", lambda _frame, _cfg: (None, features, None, None)
    )

    result = promote.gate2_staging(
        {"name": "churn", "model_name": "churn", "threshold": 0.5}
    )

    assert result.decision.accepted
    client.get_run.assert_not_called()


def test_gate2_accepts_objective_checks_and_records_informational_results(monkeypatch):
    client = Mock()
    aliases = {
        promote.STAGING: _version("2", "staging-run"),
        promote.PRODUCTION: _version("1", "production-run"),
    }
    features = pd.DataFrame(np.zeros((4, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)
    monkeypatch.setattr(promote, "_load_sklearn_model", lambda _name, _version: ProbabilityModel())
    monkeypatch.setattr(promote, "_signature_fields", lambda _name, _version: FEATURE_COLUMNS)
    monkeypatch.setattr(promote.churn_data, "load_data", lambda _cfg: pd.DataFrame())
    monkeypatch.setattr(
        promote.churn_data,
        "split_data",
        lambda _frame, _cfg: (None, features, None, None),
    )

    result = promote.gate2_staging(
        {"name": "churn", "model_name": "churn", "threshold": 0.5}
    )

    assert result.decision.accepted
    assert result.observations["sample_count"] == 4.0
    assert "informational only" in result.decision.summary()
    client.set_model_version_tag.assert_any_call(
        "churn", "2", promote.GATE2_BASELINE_TAG, "1"
    )


def test_promotion_updates_previous_then_production(monkeypatch):
    client = Mock()
    staged = _version(
        "2", "staging-run", {promote.GATE1_TAG: "accepted", promote.GATE2_TAG: "accepted", promote.GATE2_BASELINE_TAG: "1"}
    )
    aliases = {
        promote.STAGING: staged,
        promote.PRODUCTION: _version("1", "production-run"),
        promote.PREVIOUS: _version("0", "previous-run"),
    }
    client.get_model_version.return_value = staged
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)

    result = promote.promote_staging("churn")

    assert result.decision.accepted
    client.set_registered_model_alias.assert_any_call("churn", "previous_production", "1")
    client.set_registered_model_alias.assert_any_call("churn", "production", "2")


def test_promotion_failure_restores_production_and_previous_aliases(monkeypatch):
    client = Mock()
    staged = _version(
        "2", "staging-run", {promote.GATE1_TAG: "accepted", promote.GATE2_TAG: "accepted", promote.GATE2_BASELINE_TAG: "1"}
    )
    aliases = {
        promote.STAGING: staged,
        promote.PRODUCTION: _version("1", "production-run"),
        promote.PREVIOUS: _version("0", "previous-run"),
    }
    client.get_model_version.return_value = staged
    client.set_registered_model_alias.side_effect = [
        None,
        MlflowException("write failed"),
        None,
        None,
    ]
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)

    with pytest.raises(promote.RegistryOperationError, match="alias promotion failed"):
        promote.promote_staging("churn")

    calls = client.set_registered_model_alias.call_args_list
    assert [call.args[1:] for call in calls] == [
        ("previous_production", "1"),
        ("production", "2"),
        ("production", "1"),
        ("previous_production", "0"),
    ]


def test_repeated_promotion_is_idempotent(monkeypatch):
    client = Mock()
    already_production = _version(
        "2", "run", {promote.GATE1_TAG: "accepted", promote.GATE2_TAG: "accepted"}
    )
    aliases = {promote.STAGING: already_production, promote.PRODUCTION: already_production}
    client.get_model_version.return_value = already_production
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)

    first = promote.promote_staging("churn")
    second = promote.promote_staging("churn")

    assert first.decision.accepted and second.decision.accepted
    client.set_registered_model_alias.assert_not_called()


def test_promotion_rejects_stale_gate2_baseline_without_alias_changes(monkeypatch):
    client = Mock()
    staged = _version(
        "2", "staging-run", {promote.GATE1_TAG: "accepted", promote.GATE2_TAG: "accepted", promote.GATE2_BASELINE_TAG: "0"}
    )
    aliases = {promote.STAGING: staged, promote.PRODUCTION: _version("1", "production-run")}
    client.get_model_version.return_value = staged
    monkeypatch.setattr(promote, "_client", lambda: client)
    _patch_aliases(monkeypatch, aliases)

    result = promote.promote_staging("churn")

    assert not result.decision.accepted
    assert "baseline is stale" in result.decision.summary()
    client.set_registered_model_alias.assert_not_called()
