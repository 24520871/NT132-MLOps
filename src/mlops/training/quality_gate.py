"""Quality gate: decide whether a candidate model may become production. Pure logic, no MLflow."""
import math
from dataclasses import dataclass, field
from numbers import Real


@dataclass
class GateDecision:
    accepted: bool
    reasons: list[str] = field(default_factory=list)  # why it failed, or why it passed

    def summary(self) -> str:
        return ("ACCEPTED: " if self.accepted else "REJECTED: ") + "; ".join(self.reasons)


def evaluate_gate(
    candidate: dict[str, float],
    gate_cfg: dict,
    production: dict[str, float] | None = None,
) -> GateDecision:
    """Check required Gate 1 metrics and compare AUC with production when available."""
    failures: list[str] = []
    required_metrics = ("auc", "recall")
    metric_values: dict[str, float] = {}
    for name in required_metrics:
        value = candidate.get(name)
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
            failures.append(f"candidate metric '{name}' is missing or invalid")
        else:
            metric_values[name] = float(value)

    thresholds: dict[str, float] = {}
    for name in ("min_auc", "min_recall"):
        value = gate_cfg.get(name)
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
            failures.append(f"Gate 1 threshold '{name}' is missing or invalid")
        else:
            thresholds[name] = float(value)

    if failures:
        return GateDecision(False, failures)

    auc, recall = metric_values["auc"], metric_values["recall"]
    min_auc, min_recall = thresholds["min_auc"], thresholds["min_recall"]

    if not 0.0 <= auc <= 1.0:
        failures.append(f"candidate metric 'auc' must be between 0 and 1, got {auc}")
    if not 0.0 <= recall <= 1.0:
        failures.append(f"candidate metric 'recall' must be between 0 and 1, got {recall}")
    if not 0.0 <= min_auc <= 1.0:
        failures.append(f"Gate 1 threshold 'min_auc' must be between 0 and 1, got {min_auc}")
    if not 0.0 <= min_recall <= 1.0:
        failures.append(f"Gate 1 threshold 'min_recall' must be between 0 and 1, got {min_recall}")
    if auc < min_auc:
        failures.append(f"auc {auc:.4f} < min_auc {min_auc}")
    if recall < min_recall:
        failures.append(f"recall {recall:.4f} < min_recall {min_recall}")
    if production is not None:
        prod_auc = production.get("auc")
        max_drop = gate_cfg.get("max_auc_drop_vs_production")
        if not isinstance(prod_auc, Real) or isinstance(prod_auc, bool) or not math.isfinite(prod_auc):
            failures.append("production metric 'auc' is missing or invalid")
        elif not 0.0 <= prod_auc <= 1.0:
            failures.append(f"production metric 'auc' must be between 0 and 1, got {prod_auc}")
        elif not isinstance(max_drop, Real) or isinstance(max_drop, bool) or not math.isfinite(max_drop):
            failures.append("Gate 1 threshold 'max_auc_drop_vs_production' is missing or invalid")
        elif not 0.0 <= max_drop <= 1.0:
            failures.append(
                "Gate 1 threshold 'max_auc_drop_vs_production' must be between 0 and 1"
            )
        elif auc < float(prod_auc) - float(max_drop):
            failures.append(f"auc {auc:.4f} worse than production {float(prod_auc):.4f}")

    if failures:
        return GateDecision(False, failures)
    note = "meets thresholds" + (" and is not worse than production" if production else "; no production model yet")
    return GateDecision(True, [note])
