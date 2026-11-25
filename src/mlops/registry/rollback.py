"""Restore the previous production model through the MLflow Model Registry.

CLI: ``python -m mlops.registry.rollback churn [--reason ...]``
"""
import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

from mlflow.exceptions import MlflowException

from mlops.config import load_config
from mlops.registry.promote import PREVIOUS, PRODUCTION, _client, _version_by_alias

log = logging.getLogger(__name__)

ROLLED_BACK = "rolled_back"
ROLLBACK_REASON = "rollback_reason"
LAST_ROLLBACK_AT = "last_rollback_at"


class RollbackError(RuntimeError):
    """The registry operation failed or rollback safeguards rejected it."""


@dataclass(frozen=True)
class RollbackResult:
    rolled_back: bool
    failed_version: str | None
    restored_version: str | None
    warning: str | None = None


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RollbackError(f"Invalid registry tag {LAST_ROLLBACK_AT}={value!r}") from exc
    if parsed.tzinfo is None:
        raise RollbackError(f"Registry tag {LAST_ROLLBACK_AT} must include a timezone")
    return parsed


def rollback_model(
    model_name: str,
    reason: str,
    cooldown_seconds: int | None = None,
) -> RollbackResult:
    """Move production to previous_production, applying configured safeguards.

    ``cooldown_seconds`` is optional because the project has not chosen a value yet.
    When supplied, the timestamp is stored on the registered model to enforce cooldown
    across rollback calls. A version tagged as rolled back can never be rolled back again.
    """
    if not reason.strip():
        raise ValueError("Rollback reason must not be empty")
    if cooldown_seconds is not None and cooldown_seconds < 0:
        raise ValueError("cooldown_seconds must be zero or greater")

    try:
        client = _client()
        production = _version_by_alias(client, model_name, PRODUCTION)
        if production is None:
            raise RollbackError(f"No '{PRODUCTION}' model for '{model_name}'; nothing to roll back")
        previous = _version_by_alias(client, model_name, PREVIOUS)
        if previous is None:
            warning = (
                f"No '{PREVIOUS}' version for '{model_name}'; production was left unchanged. "
                "Alert-only rollback required."
            )
            log.warning(warning)
            return RollbackResult(False, str(production.version), None, warning)

        failed_version = str(production.version)
        restored_version = str(previous.version)
        if failed_version == restored_version:
            warning = (
                f"Production and '{PREVIOUS}' both point to v{failed_version}; "
                "no alias change was made."
            )
            log.warning(warning)
            return RollbackResult(False, failed_version, restored_version, warning)

        tags = client.get_model_version(model_name, failed_version).tags
        if tags.get(ROLLED_BACK, "").lower() == "true":
            raise RollbackError(
                f"Production v{failed_version} was already rolled back; repeated rollback is blocked"
            )

        registered_model = client.get_registered_model(model_name)
        last_rollback_at = registered_model.tags.get(LAST_ROLLBACK_AT)
        now = datetime.now(UTC)
        if cooldown_seconds is not None and last_rollback_at:
            last = _parse_timestamp(last_rollback_at)
            elapsed = (now - last).total_seconds()
            if elapsed < cooldown_seconds:
                remaining = cooldown_seconds - elapsed
                raise RollbackError(
                    f"Rollback cooldown is active for '{model_name}' ({remaining:.0f}s remaining)"
                )

        old_reason = tags.get(ROLLBACK_REASON)
        old_rollback_at = registered_model.tags.get(LAST_ROLLBACK_AT)
        try:
            client.set_model_version_tag(model_name, failed_version, ROLLBACK_REASON, reason.strip())
            if cooldown_seconds is not None:
                client.set_registered_model_tag(model_name, LAST_ROLLBACK_AT, now.isoformat())
            client.set_registered_model_alias(model_name, PRODUCTION, restored_version)
            client.set_model_version_tag(model_name, failed_version, ROLLED_BACK, "true")
        except MlflowException as operation_error:
            cleanup_errors = []
            try:
                client.set_registered_model_alias(model_name, PRODUCTION, failed_version)
            except MlflowException as cleanup_error:
                cleanup_errors.append(f"could not restore production alias: {cleanup_error}")
            for key, old_value in ((ROLLED_BACK, tags.get(ROLLED_BACK)), (ROLLBACK_REASON, old_reason)):
                try:
                    if old_value is None:
                        client.delete_model_version_tag(model_name, failed_version, key)
                    else:
                        client.set_model_version_tag(model_name, failed_version, key, old_value)
                except MlflowException as cleanup_error:
                    cleanup_errors.append(f"could not restore {key} tag: {cleanup_error}")
            if cooldown_seconds is not None:
                try:
                    if old_rollback_at is None:
                        client.delete_registered_model_tag(model_name, LAST_ROLLBACK_AT)
                    else:
                        client.set_registered_model_tag(model_name, LAST_ROLLBACK_AT, old_rollback_at)
                except MlflowException as cleanup_error:
                    cleanup_errors.append(f"could not restore {LAST_ROLLBACK_AT}: {cleanup_error}")
            detail = f"MLflow registry error during rollback of '{model_name}': {operation_error}"
            if cleanup_errors:
                detail += "; compensation incomplete, registry needs review: " + "; ".join(cleanup_errors)
            raise RollbackError(detail) from operation_error

        if cooldown_seconds is None:
            log.warning("Rollback cooldown is not enforced: no cooldown_seconds value is configured")

        log.warning(
            "Rolled back %s from v%s to v%s: %s",
            model_name,
            failed_version,
            restored_version,
            reason.strip(),
        )
        return RollbackResult(True, failed_version, restored_version)
    except MlflowException as exc:
        log.exception("MLflow registry error during rollback of %s", model_name)
        raise RollbackError(f"MLflow registry error during rollback of '{model_name}': {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore the previous production model")
    parser.add_argument("usecase", help="e.g. churn")
    parser.add_argument("--reason", default="manual rollback", help="why the production model is being rolled back")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.usecase)
    rollback_cfg = cfg.get("rollback", {})
    try:
        result = rollback_model(
            cfg["model_name"],
            args.reason,
            rollback_cfg.get("cooldown_seconds"),
        )
    except (RollbackError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(2)
    if result.rolled_back:
        print(f"Rolled back v{result.failed_version} to v{result.restored_version}")
        sys.exit(0)
    print(result.warning)
    sys.exit(1)


if __name__ == "__main__":
    main()
