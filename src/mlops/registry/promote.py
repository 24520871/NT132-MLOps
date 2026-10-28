"""MLflow model registry helpers: register candidates, gate and promote, never silently.

CLI:  python -m mlops.registry.promote churn     (exit code 0 = promoted, 1 = rejected)
"""
import argparse
import logging
import sys
from dataclasses import dataclass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlops.config import load_config, tracking_uri
from mlops.training.quality_gate import GateDecision, evaluate_gate

log = logging.getLogger(__name__)

CANDIDATE, PRODUCTION, PREVIOUS = "candidate", "production", "previous_production"


class NoCandidateError(RuntimeError):
    """No model version carries the `candidate` alias."""


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


def promote_candidate(model_name: str, gate_cfg: dict) -> PromotionResult:
    """Run the quality gate on the candidate. Only an accepted candidate becomes `production`."""
    client = _client()
    cand = _version_by_alias(client, model_name, CANDIDATE)
    if cand is None:
        raise NoCandidateError(
            f"No '{CANDIDATE}' version of '{model_name}'. Run training first "
            "(python -m usecases.churn.train)."
        )
    prod = _version_by_alias(client, model_name, PRODUCTION)
    prod_metrics = _run_metrics(client, prod.run_id) if prod else None

    decision = evaluate_gate(_run_metrics(client, cand.run_id), gate_cfg, prod_metrics)
    client.set_model_version_tag(model_name, cand.version, "gate", "accepted" if decision.accepted else "rejected")
    client.set_model_version_tag(model_name, cand.version, "gate_reasons", "; ".join(decision.reasons))

    if decision.accepted:
        if prod:
            client.set_registered_model_alias(model_name, PREVIOUS, prod.version)
        client.set_registered_model_alias(model_name, PRODUCTION, cand.version)
        log.info("Promoted %s v%s to production", model_name, cand.version)
    else:
        log.warning("Rejected %s v%s: %s", model_name, cand.version, decision.summary())
    client.delete_registered_model_alias(model_name, CANDIDATE)  # decision made either way
    return PromotionResult(decision, str(cand.version), str(prod.version) if prod else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate and promote the candidate model")
    parser.add_argument("usecase", help="e.g. churn")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.usecase)
    try:
        result = promote_candidate(cfg["model_name"], cfg["gate"])
    except NoCandidateError as exc:
        print(exc)
        sys.exit(2)
    print(f"v{result.version}: {result.decision.summary()}")
    sys.exit(0 if result.decision.accepted else 1)


if __name__ == "__main__":
    main()
