"""Run Gate 1, Gate 2, and the guarded MLflow alias promotion as separate stages."""
import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from numbers import Real
from typing import Sequence

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlops.config import load_config, tracking_uri
from mlops.training.quality_gate import GateDecision, evaluate_gate
from mlops.training.staging_check import evaluate_gate2
from usecases.churn import data as churn_data
from usecases.churn.schema import FEATURE_COLUMNS

log = logging.getLogger(__name__)

CANDIDATE, STAGING, PRODUCTION, PREVIOUS = (
    "candidate",
    "staging",
    "production",
    "previous_production",
)
GATE1_TAG = "gate1"
GATE2_TAG = "gate2"
GATE_REASONS_TAG = "gate_reasons"
GATE2_REASONS_TAG = "gate2_reasons"
GATE2_BASELINE_TAG = "gate2_production_version"
GATE2_OBSERVATIONS_TAG = "gate2_observations"


class NoCandidateError(RuntimeError):
    """No model version carries the `candidate` alias."""


class NoStagingError(RuntimeError):
    """No model version carries the `staging` alias."""


class RegistryOperationError(RuntimeError):
    """An MLflow operation failed or a registry compensation was incomplete."""


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri())


def _version_by_alias(client: MlflowClient, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException as exc:
        missing_alias = (
            exc.error_code == "INVALID_PARAMETER_VALUE"
            and f"alias {alias} not found" in exc.message.lower()
        )
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST" or missing_alias:
            return None
        raise


def _run_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    try:
        return dict(client.get_run(run_id).data.metrics)
    except MlflowException as exc:
        raise RegistryOperationError(f"Could not read metrics for MLflow run {run_id}: {exc}") from exc


def _metric_failures(
    metrics: dict[str, float], required: tuple[str, ...], label: str
) -> list[str]:
    failures: list[str] = []
    for name in required:
        value = metrics.get(name)
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
            failures.append(f"{label} metric '{name}' is missing or invalid")
        elif not 0.0 <= float(value) <= 1.0:
            failures.append(f"{label} metric '{name}' must be between 0 and 1, got {value}")
    return failures


def _version_metrics(
    client: MlflowClient,
    model_version,
    required: tuple[str, ...],
    label: str,
) -> tuple[dict[str, float], list[str]]:
    if not model_version.run_id:
        return {}, [f"{label} model version v{model_version.version} has no MLflow run_id"]
    metrics = _run_metrics(client, model_version.run_id)
    return metrics, _metric_failures(metrics, required, label)


def register_candidate(model_name: str, run_id: str) -> str:
    """Register the model logged in a run and mark that version `candidate`."""
    mlflow.set_tracking_uri(tracking_uri())
    try:
        model_version = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        _client().set_registered_model_alias(model_name, CANDIDATE, model_version.version)
    except MlflowException as exc:
        raise RegistryOperationError(f"Could not register candidate for '{model_name}': {exc}") from exc
    return str(model_version.version)


@dataclass(frozen=True)
class PromotionResult:
    stage: str
    decision: GateDecision
    version: str
    previous_production: str | None = None
    observations: dict[str, float] = field(default_factory=dict)


def _tag_gate1(client: MlflowClient, model_name: str, version: str, decision: GateDecision) -> None:
    status = "accepted" if decision.accepted else "rejected"
    try:
        client.set_model_version_tag(model_name, version, GATE1_TAG, status)
        # Keep the existing tag contract for dashboards and older tooling.
        client.set_model_version_tag(model_name, version, "gate", status)
        client.set_model_version_tag(model_name, version, GATE_REASONS_TAG, "; ".join(decision.reasons))
    except MlflowException as exc:
        raise RegistryOperationError(f"Could not record Gate 1 result for v{version}: {exc}") from exc


def gate1_candidate(model_name: str, gate_cfg: dict) -> PromotionResult:
    """Evaluate candidate metrics and move an accepted candidate to `staging` only."""
    client = _client()
    candidate = _version_by_alias(client, model_name, CANDIDATE)
    if candidate is None:
        staged = _version_by_alias(client, model_name, STAGING)
        if staged is not None:
            tags = client.get_model_version(model_name, staged.version).tags
            if tags.get(GATE1_TAG) == "accepted":
                production = _version_by_alias(client, model_name, PRODUCTION)
                return PromotionResult(
                    GATE1_TAG,
                    GateDecision(True, ["candidate was already accepted into staging"]),
                    str(staged.version),
                    str(production.version) if production is not None else None,
                )
        raise NoCandidateError(
            f"No '{CANDIDATE}' version of '{model_name}'. Train first with "
            f"python -m usecases.{model_name}.train."
        )

    production = _version_by_alias(client, model_name, PRODUCTION)
    candidate_metrics, failures = _version_metrics(
        client, candidate, ("auc", "recall"), "candidate"
    )
    production_metrics: dict[str, float] | None = None
    if production is not None:
        production_metrics, production_failures = _version_metrics(
            client, production, ("auc",), "production"
        )
        failures.extend(production_failures)

    decision = (
        GateDecision(False, failures)
        if failures
        else evaluate_gate(candidate_metrics, gate_cfg, production_metrics)
    )
    _tag_gate1(client, model_name, str(candidate.version), decision)
    if decision.accepted:
        try:
            client.set_registered_model_alias(model_name, STAGING, candidate.version)
            client.delete_registered_model_alias(model_name, CANDIDATE)
        except MlflowException as exc:
            raise RegistryOperationError(
                f"Gate 1 accepted v{candidate.version}, but staging alias update failed: {exc}"
            ) from exc
        log.info("Gate 1 accepted %s v%s into staging", model_name, candidate.version)
    else:
        log.warning("Gate 1 rejected %s v%s: %s", model_name, candidate.version, decision.summary())
    return PromotionResult(
        GATE1_TAG,
        decision,
        str(candidate.version),
        str(production.version) if production is not None else None,
    )


def _signature_fields(model_name: str, version: str) -> list[str] | None:
    uri = f"models:/{model_name}/{version}"
    try:
        info = mlflow.models.get_model_info(uri)
    except MlflowException as exc:
        raise RegistryOperationError(f"Could not read MLflow model signature for v{version}: {exc}") from exc
    if info.signature is None:
        return None
    return [field.name for field in info.signature.inputs]


def _load_sklearn_model(model_name: str, version: str):
    uri = f"models:/{model_name}/{version}"
    try:
        return mlflow.sklearn.load_model(uri)
    except MlflowException as exc:
        raise RegistryOperationError(f"Could not load MLflow model v{version}: {exc}") from exc


def _tag_gate2(
    client: MlflowClient,
    model_name: str,
    version: str,
    decision: GateDecision,
    production_version: str | None,
    observations: dict[str, float],
) -> None:
    status = "accepted" if decision.accepted else "rejected"
    try:
        client.set_model_version_tag(model_name, version, GATE2_TAG, status)
        client.set_model_version_tag(
            model_name, version, GATE2_REASONS_TAG, "; ".join(decision.reasons)
        )
        if decision.accepted and production_version is not None:
            client.set_model_version_tag(
                model_name, version, GATE2_BASELINE_TAG, production_version
            )
            client.set_model_version_tag(
                model_name,
                version,
                GATE2_OBSERVATIONS_TAG,
                json.dumps(observations, sort_keys=True),
            )
    except MlflowException as exc:
        raise RegistryOperationError(f"Could not record Gate 2 result for v{version}: {exc}") from exc


def gate2_staging(cfg: dict) -> PromotionResult:
    """Run objective checks on the staged model; compare and latency are informational."""
    if cfg.get("name") != "churn":
        raise RegistryOperationError("Gate 2 currently supports only the configured churn use case")

    client = _client()
    model_name = cfg["model_name"]
    staged = _version_by_alias(client, model_name, STAGING)
    if staged is None:
        raise NoStagingError(f"No '{STAGING}' version of '{model_name}' is available for Gate 2")
    production = _version_by_alias(client, model_name, PRODUCTION)
    if production is None:
        decision = GateDecision(False, ["production baseline is required for Gate 2; no promotion made"])
        _tag_gate2(client, model_name, str(staged.version), decision, None, {})
        return PromotionResult(GATE2_TAG, decision, str(staged.version))

    observations: dict[str, float] = {}
    mlflow.set_tracking_uri(tracking_uri())
    try:
        staged_model = _load_sklearn_model(model_name, str(staged.version))
        production_model = _load_sklearn_model(model_name, str(production.version))
        staged_signature = _signature_fields(model_name, str(staged.version))
        production_signature = _signature_fields(model_name, str(production.version))
        frame = churn_data.load_data(cfg)
        _, held_out, _, _ = churn_data.split_data(frame, cfg)
    except (OSError, ValueError, KeyError) as exc:
        decision = GateDecision(False, [f"could not prepare Gate 2 checks: {exc}"])
    else:
        result = evaluate_gate2(
            staged_model,
            production_model,
            held_out,
            FEATURE_COLUMNS,
            staged_signature,
            production_signature,
            float(cfg["threshold"]),
        )
        decision = GateDecision(result.passed, result.reasons)
        observations = result.observations

    current_production = _version_by_alias(client, model_name, PRODUCTION)
    if current_production is None or str(current_production.version) != str(production.version):
        decision = GateDecision(False, ["production baseline changed during Gate 2; rerun the check"])
        observations = {}
    _tag_gate2(
        client,
        model_name,
        str(staged.version),
        decision,
        str(production.version) if decision.accepted else None,
        observations,
    )
    if decision.accepted:
        log.info(
            "Gate 2 passed for %s v%s against production v%s; observations=%s",
            model_name,
            staged.version,
            production.version,
            observations,
        )
    else:
        log.warning("Gate 2 rejected %s v%s: %s", model_name, staged.version, decision.summary())
    return PromotionResult(
        GATE2_TAG,
        decision,
        str(staged.version),
        str(production.version),
        observations,
    )


def _restore_alias(
    client: MlflowClient, model_name: str, alias: str, version: str | None
) -> None:
    if version is None:
        if _version_by_alias(client, model_name, alias) is not None:
            client.delete_registered_model_alias(model_name, alias)
    else:
        client.set_registered_model_alias(model_name, alias, version)


def promote_staging(model_name: str) -> PromotionResult:
    """Promote only a staged version with accepted Gate 1/2 and a current baseline."""
    client = _client()
    staged = _version_by_alias(client, model_name, STAGING)
    if staged is None:
        raise NoStagingError(f"No '{STAGING}' version of '{model_name}' is available to promote")
    production = _version_by_alias(client, model_name, PRODUCTION)
    tags = client.get_model_version(model_name, staged.version).tags

    if production is not None and str(staged.version) == str(production.version):
        if tags.get(GATE1_TAG) != "accepted" or tags.get(GATE2_TAG) != "accepted":
            decision = GateDecision(False, ["already-production version has no accepted Gate 1/2 tags"])
            return PromotionResult("promote", decision, str(staged.version), str(production.version))
        decision = GateDecision(True, ["staged version is already production; no alias changes made"])
        return PromotionResult("promote", decision, str(staged.version), str(production.version))
    if production is None:
        decision = GateDecision(False, ["production baseline is required; no promotion made"])
        return PromotionResult("promote", decision, str(staged.version))
    if tags.get(GATE1_TAG) != "accepted":
        decision = GateDecision(False, ["staged version has no accepted Gate 1 result"])
        return PromotionResult("promote", decision, str(staged.version), str(production.version))
    if tags.get(GATE2_TAG) != "accepted":
        decision = GateDecision(False, ["staged version has no accepted Gate 2 result"])
        return PromotionResult("promote", decision, str(staged.version), str(production.version))
    if tags.get(GATE2_BASELINE_TAG) != str(production.version):
        decision = GateDecision(False, ["Gate 2 baseline is stale; rerun Gate 2 before promotion"])
        return PromotionResult("promote", decision, str(staged.version), str(production.version))

    previous = _version_by_alias(client, model_name, PREVIOUS)
    previous_version = str(previous.version) if previous is not None else None
    try:
        client.set_registered_model_alias(model_name, PREVIOUS, production.version)
        client.set_registered_model_alias(model_name, PRODUCTION, staged.version)
    except MlflowException as operation_error:
        cleanup_errors: list[str] = []
        for alias, version in ((PRODUCTION, str(production.version)), (PREVIOUS, previous_version)):
            try:
                _restore_alias(client, model_name, alias, version)
            except MlflowException as cleanup_error:
                cleanup_errors.append(f"could not restore alias '{alias}': {cleanup_error}")
        detail = f"MLflow alias promotion failed for '{model_name}' v{staged.version}: {operation_error}"
        if cleanup_errors:
            detail += "; registry compensation incomplete: " + "; ".join(cleanup_errors)
        raise RegistryOperationError(detail) from operation_error

    decision = GateDecision(True, ["Gate 1 and Gate 2 passed; production alias updated"])
    log.info(
        "Promoted %s v%s to production; previous_production=v%s",
        model_name,
        staged.version,
        production.version,
    )
    return PromotionResult("promote", decision, str(staged.version), str(production.version))


def _run_stage(usecase: str, stage: str) -> PromotionResult:
    cfg = load_config(usecase)
    if stage == "gate1":
        return gate1_candidate(cfg["model_name"], cfg["gate"])
    if stage == "gate2":
        return gate2_staging(cfg)
    return promote_staging(cfg["model_name"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a model quality gate or guarded promotion")
    parser.add_argument("usecase", help="e.g. churn")
    parser.add_argument(
        "--stage",
        choices=("gate1", "gate2", "promote"),
        default="promote",
        help="gate1 evaluates candidate into staging; gate2 validates staging; promote updates aliases",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        result = _run_stage(args.usecase, args.stage)
    except (NoCandidateError, NoStagingError) as exc:
        log.error("%s", exc)
        return 2
    except (RegistryOperationError, MlflowException, KeyError, OSError, ValueError) as exc:
        log.exception("Stage '%s' could not complete", args.stage)
        print(str(exc))
        return 2

    print(f"{args.stage} v{result.version}: {result.decision.summary()}")
    if result.observations:
        print(f"informational observations: {json.dumps(result.observations, sort_keys=True)}")
    return 0 if result.decision.accepted else 1


if __name__ == "__main__":
    sys.exit(main())
