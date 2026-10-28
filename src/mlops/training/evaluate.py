"""Classification metrics shared by all classification use cases."""
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(y_true, proba, threshold: float = 0.5) -> tuple[dict, dict]:
    """Return (metrics, confusion matrix counts) for positive-class probabilities."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return metrics, {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
