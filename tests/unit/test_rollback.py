from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from mlops.registry import rollback


@pytest.fixture
def registry(monkeypatch):
    client = Mock()
    production = SimpleNamespace(version="2")
    previous = SimpleNamespace(version="1")
    client.get_model_version.return_value = SimpleNamespace(tags={})
    client.get_registered_model.return_value = SimpleNamespace(tags={})
    aliases = {"production": production, "previous_production": previous}
    monkeypatch.setattr(rollback, "_client", lambda: client)
    monkeypatch.setattr(rollback, "_version_by_alias", lambda _client, _name, alias: aliases.get(alias))
    return client, aliases


def test_rollback_restores_previous_and_tags_failed_version(registry):
    client, _ = registry

    result = rollback.rollback_model("churn", "latency regression")

    assert result.rolled_back
    assert result.failed_version == "2"
    assert result.restored_version == "1"
    client.set_registered_model_alias.assert_called_once_with("churn", "production", "1")
    client.set_model_version_tag.assert_any_call("churn", "2", "rolled_back", "true")
    client.set_model_version_tag.assert_any_call("churn", "2", "rollback_reason", "latency regression")


def test_missing_previous_version_is_alert_only(registry, monkeypatch):
    client, aliases = registry
    aliases["previous_production"] = None

    result = rollback.rollback_model("churn", "manual request")

    assert not result.rolled_back
    assert "Alert-only" in result.warning
    client.set_registered_model_alias.assert_not_called()
    client.set_model_version_tag.assert_not_called()


def test_same_production_and_previous_alias_is_a_noop(registry):
    client, aliases = registry
    aliases["previous_production"] = aliases["production"]

    result = rollback.rollback_model("churn", "repeat call")

    assert not result.rolled_back
    assert "both point" in result.warning
    client.set_registered_model_alias.assert_not_called()
    client.set_model_version_tag.assert_not_called()


def test_already_rolled_back_version_is_blocked(registry):
    client, _ = registry
    client.get_model_version.return_value = SimpleNamespace(tags={"rolled_back": "true"})

    with pytest.raises(rollback.RollbackError, match="already rolled back"):
        rollback.rollback_model("churn", "repeat")

    client.set_registered_model_alias.assert_not_called()


def test_cooldown_blocks_rollback(registry):
    client, _ = registry
    now = datetime.now(UTC)
    client.get_registered_model.return_value = SimpleNamespace(
        tags={"last_rollback_at": now.isoformat()}
    )

    with pytest.raises(rollback.RollbackError, match="cooldown is active"):
        rollback.rollback_model("churn", "too soon", cooldown_seconds=3600)

    client.set_registered_model_alias.assert_not_called()


def test_cooldown_timestamp_is_recorded_when_configured(registry):
    client, _ = registry

    result = rollback.rollback_model("churn", "operator", cooldown_seconds=0)

    assert result.rolled_back
    client.set_registered_model_tag.assert_called_once()
    assert client.set_registered_model_tag.call_args.args[:2] == (
        "churn",
        "last_rollback_at",
    )


def test_registry_errors_are_reported_safely(monkeypatch):
    monkeypatch.setattr(rollback, "_client", Mock(side_effect=MlflowException("registry unavailable")))

    with pytest.raises(rollback.RollbackError, match="MLflow registry error"):
        rollback.rollback_model("churn", "registry failure")


def test_tag_failure_restores_production_alias(registry):
    client, _ = registry
    client.set_model_version_tag.side_effect = [None, MlflowException("tag write failed"), None]

    with pytest.raises(rollback.RollbackError, match="tag write failed"):
        rollback.rollback_model("churn", "tag failure")

    calls = client.set_registered_model_alias.call_args_list
    assert len(calls) == 2
    assert calls[0].args == ("churn", "production", "1")
    assert calls[1].args == ("churn", "production", "2")


def test_empty_reason_is_rejected_before_registry_access(monkeypatch):
    client_factory = Mock()
    monkeypatch.setattr(rollback, "_client", client_factory)

    with pytest.raises(ValueError, match="must not be empty"):
        rollback.rollback_model("churn", "  ")

    client_factory.assert_not_called()
