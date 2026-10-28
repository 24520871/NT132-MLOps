"""Model definitions: logistic regression baseline and XGBoost, both full sklearn Pipelines."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from usecases.churn.features import build_preprocessor


def build_models(cfg: dict, y_train) -> dict[str, Pipeline]:
    """Return {name: untrained pipeline}. Class imbalance handled via weights (train labels only)."""
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    seed = cfg["split"]["seed"]
    return {
        "logreg": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(class_weight="balanced", random_state=seed, **cfg["models"]["logreg"]),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        scale_pos_weight=pos_weight,
                        random_state=seed,
                        eval_metric="logloss",
                        n_jobs=1,
                        **cfg["models"]["xgboost"],
                    ),
                ),
            ]
        ),
    }
