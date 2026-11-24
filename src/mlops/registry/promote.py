"""MLflow model registry helpers: register candidates, gate and promote, never silently.

Lifecycle:  candidate --Gate 1 (offline metrics)--> staging --Gate 2 (automatic staging check)--> production

CLI:  python -m mlops.registry.promote churn [--stage gate1|gate2|all]
      exit code 0 = passed, 1 = rejected, 2 = nothing to promote (no candidate / no staging version)
"""
import argparse
import importlib
import logging
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlops.config import load_config, tracking_uri
from mlops.training.quality_gate import GateDecision, evaluate_gate
from mlops.training.staging_check import evaluate_staging

log = logging.getLogger(__name__)

CANDIDATE, STAGING, PRODUCTION, PREVIOUS = "candidate", "staging", "production", "previous_production"


class NoCandidateError(RuntimeError):
    """No model version carries the `candidate` alias."""


class NoStagingError(RuntimeError):
    """No model version carries the `staging` alias."""


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri())


def _version_by_alias(client: MlflowClient, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def _run_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    return dict(client.get_run(run_id).data.metrics)


def register_candidate(model_name: str, run_id: str) -> str:
    """Register the model logged in a run and mark that version `candidate`. Returns the version."""
    mlflow.set_tracking_uri(tracking_uri())
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    _client().set_registered_model_alias(model_name, CANDIDATE, mv.version)
    return str(mv.version)


@dataclass
class PromotionResult:
    decision: GateDecision
    version: str
    previous_production: str | None


def stage_candidate(model_name: str, gate_cfg: dict) -> PromotionResult:
    """Gate 1: run the quality gate on the candidate. Only an accepted candidate becomes `staging`."""
    client = _client()
    cand = _version_by_alias(client, model_name, CANDIDATE)
    if cand is None:
        raise NoCandidateError(
            f"No '{CANDIDATE}' version of '{model_name}'. Run training first "
            f"(python -m usecases.{model_name}.train)."
        )
    prod = _version_by_alias(client, model_name, PRODUCTION)
    prod_metrics = _run_metrics(client, prod.run_id) if prod else None

    decision = evaluate_gate(_run_metrics(client, cand.run_id), gate_cfg, prod_metrics)
    client.set_model_version_tag(model_name, cand.version, "gate", "accepted" if decision.accepted else "rejected")
    client.set_model_version_tag(model_name, cand.version, "gate_reasons", "; ".join(decision.reasons))

    if decision.accepted:
        client.set_registered_model_alias(model_name, STAGING, cand.version)
        log.info("Gate 1 passed: %s v%s is now staging", model_name, cand.version)
    else:
        log.warning("Gate 1 rejected %s v%s: %s", model_name, cand.version, decision.summary())
    client.delete_registered_model_alias(model_name, CANDIDATE)  # decision made either way
    return PromotionResult(decision, str(cand.version), str(prod.version) if prod else None)


def promote_staging(
    model_name: str, staging_cfg: dict, sample: pd.DataFrame, threshold: float = 0.5
) -> PromotionResult:
    """Gate 2: check the staging model on `sample`. Only an accepted version becomes `production`."""
    client = _client()
    stg = _version_by_alias(client, model_name, STAGING)
    if stg is None:
        raise NoStagingError(
            f"No '{STAGING}' version of '{model_name}'. Run Gate 1 first "
            f"(python -m mlops.registry.promote {model_name} --stage gate1)."
        )
    prod = _version_by_alias(client, model_name, PRODUCTION)

    mlflow.set_tracking_uri(tracking_uri())
    uri = f"models:/{model_name}/{stg.version}"
    signature = mlflow.models.get_model_info(uri).signature
    decision = evaluate_staging(
        mlflow.sklearn.load_model(uri),
        sample,
        staging_cfg,
        threshold,
        production_model=mlflow.sklearn.load_model(f"models:/{model_name}/{prod.version}") if prod else None,
        signature_columns=signature.inputs.input_names() if signature and signature.inputs else None,
    )
    client.set_model_version_tag(model_name, stg.version, "staging_gate", "accepted" if decision.accepted else "rejected")
    client.set_model_version_tag(model_name, stg.version, "staging_gate_reasons", "; ".join(decision.reasons))

    if decision.accepted:
        if prod:
            client.set_registered_model_alias(model_name, PREVIOUS, prod.version)
        client.set_registered_model_alias(model_name, PRODUCTION, stg.version)
        log.info("Gate 2 passed: %s v%s is now production", model_name, stg.version)
    else:
        log.warning("Gate 2 rejected %s v%s: %s", model_name, stg.version, decision.summary())
    client.delete_registered_model_alias(model_name, STAGING)  # decision made either way
    return PromotionResult(decision, str(stg.version), str(prod.version) if prod else None)


def _held_out_sample(usecase: str, cfg: dict) -> pd.DataFrame:
    """A reproducible sample of the held-out test split (never the training rows)."""
    mod = importlib.import_module(f"usecases.{usecase}.data")
    _, X_test, _, _ = mod.split_data(mod.load_data(cfg), cfg)
    n = min(cfg["staging_gate"]["sample_size"], len(X_test))
    return X_test.sample(n=n, random_state=cfg["split"]["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate the candidate model and promote it through staging")
    parser.add_argument("usecase", help="e.g. churn")
    parser.add_argument(
        "--stage",
        choices=["gate1", "gate2", "all"],
        default="all",
        help="gate1: candidate -> staging; gate2: staging -> production; all (default): both",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.usecase)
    try:
        if args.stage in ("gate1", "all"):
            result = stage_candidate(cfg["model_name"], cfg["gate"])
            print(f"Gate 1 v{result.version}: {result.decision.summary()}")
            if not result.decision.accepted:
                sys.exit(1)
        if args.stage in ("gate2", "all"):
            result = promote_staging(
                cfg["model_name"], cfg["staging_gate"], _held_out_sample(args.usecase, cfg), cfg["threshold"]
            )
            print(f"Gate 2 v{result.version}: {result.decision.summary()}")
            sys.exit(0 if result.decision.accepted else 1)
    except (NoCandidateError, NoStagingError) as exc:
        print(exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
