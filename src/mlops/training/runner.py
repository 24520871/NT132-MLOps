"""Generic training loop: fit each model, evaluate, log to MLflow, register the best as candidate."""
import logging
from dataclasses import dataclass

import mlflow
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from mlops.config import artifact_root, tracking_uri
from mlops.registry.promote import register_candidate
from mlops.training.evaluate import evaluate_classifier

log = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    best_model: str
    run_id: str
    version: str
    metrics: dict[str, float]
    all_metrics: dict[str, dict[str, float]]


def run_training(
    cfg: dict,
    models: dict[str, Pipeline],
    X_train,
    X_test,
    y_train,
    y_test,
    data_sha256: str = "",
) -> TrainingResult:
    """Train all models, log each as an MLflow run, register the best by AUC as `candidate`."""
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=artifact_root())
    mlflow.set_experiment(cfg["experiment"])

    runs: dict[str, tuple[str, dict[str, float]]] = {}
    for name, pipe in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics, confusion = evaluate_classifier(y_test, proba, cfg["threshold"])

            mlflow.set_tags({"usecase": cfg["name"], "model_type": name})
            mlflow.log_params(
                {
                    **{f"{name}.{k}": v for k, v in cfg["models"][name].items()},
                    "threshold": cfg["threshold"],
                    "test_size": cfg["split"]["test_size"],
                    "seed": cfg["split"]["seed"],
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "data_sha256": data_sha256,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_dict(confusion, "confusion_matrix.json")
            mlflow.sklearn.log_model(
                pipe,
                name="model",
                signature=infer_signature(X_test, proba),
                input_example=X_test.head(3),
                # skops (the default) rejects numpy/xgboost types in our pipelines; our own store is trusted
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            runs[name] = (run.info.run_id, metrics)
            log.info("%s: %s", name, {k: round(v, 4) for k, v in metrics.items()})

    best = max(runs, key=lambda n: runs[n][1]["auc"])
    run_id, metrics = runs[best]
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("selected", "true")
    version = register_candidate(cfg["model_name"], run_id)
    return TrainingResult(best, run_id, version, metrics, {n: m for n, (_, m) in runs.items()})
