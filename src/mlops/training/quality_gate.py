"""Quality gate: decide whether a candidate model may become production. Pure logic, no MLflow."""
from dataclasses import dataclass, field


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
    """Check candidate metrics against absolute thresholds and, if given, the production model."""
    failures: list[str] = []
    auc, recall = candidate["auc"], candidate["recall"]

    if auc < gate_cfg["min_auc"]:
        failures.append(f"auc {auc:.4f} < min_auc {gate_cfg['min_auc']}")
    if recall < gate_cfg["min_recall"]:
        failures.append(f"recall {recall:.4f} < min_recall {gate_cfg['min_recall']}")
    if production is not None:
        floor = production["auc"] - gate_cfg.get("max_auc_drop_vs_production", 0.0)
        if auc < floor:
            failures.append(f"auc {auc:.4f} worse than production {production['auc']:.4f}")

    if failures:
        return GateDecision(False, failures)
    note = "meets thresholds" + (" and is not worse than production" if production else "; no production model yet")
    return GateDecision(True, [note])
