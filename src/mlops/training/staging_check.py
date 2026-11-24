"""Gate 2: automatic check of the `staging` model on a held-out sample. Pure logic, no MLflow.

Gate 1 (quality_gate.py) judges offline metrics; this gate checks the model behaves sanely when actually
loaded and called: valid outputs, a plausible positive rate, not wildly different from production
(shadow comparison on the same rows) and fast enough for single-row serving.
"""
import time

import numpy as np
import pandas as pd

from mlops.training.quality_gate import GateDecision

LATENCY_CALLS = 20  # single-row predictions used to estimate p95 latency


def _positive_proba(model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(frame))[:, 1]


def _p95_latency_ms(model, sample: pd.DataFrame) -> float:
    n = min(LATENCY_CALLS, len(sample))
    times = []
    for i in range(n):
        start = time.perf_counter()
        _positive_proba(model, sample.iloc[[i]])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 95))


def evaluate_staging(
    model,
    sample: pd.DataFrame,
    staging_cfg: dict,
    threshold: float = 0.5,
    production_model=None,
    signature_columns: list[str] | None = None,
) -> GateDecision:
    """Run every Gate 2 check and report all failures at once (like Gate 1)."""
    if signature_columns is not None and set(signature_columns) != set(sample.columns):
        return GateDecision(
            False,
            [f"model signature columns {sorted(signature_columns)} != feature columns {sorted(sample.columns)}"],
        )
    try:
        proba = _positive_proba(model, sample)
    except Exception as exc:  # any failure to predict is a rejection, not a crash
        return GateDecision(False, [f"prediction failed on held-out sample: {exc}"])

    failures: list[str] = []
    facts: list[str] = []

    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        failures.append("probabilities contain NaN/inf or fall outside [0, 1]")
    else:
        rate = float((proba >= threshold).mean())
        lo, hi = staging_cfg["min_positive_rate"], staging_cfg["max_positive_rate"]
        facts.append(f"positive rate {rate:.3f}")
        if not lo <= rate <= hi:
            failures.append(f"positive rate {rate:.3f} outside [{lo}, {hi}]")
        if production_model is not None:
            diff = float(np.abs(proba - _positive_proba(production_model, sample)).mean())
            limit = staging_cfg["max_mean_abs_diff_vs_production"]
            facts.append(f"mean |dp| vs production {diff:.3f}")
            if diff > limit:
                failures.append(f"mean |dp| vs production {diff:.3f} > {limit}")

    p95 = _p95_latency_ms(model, sample)
    facts.append(f"p95 latency {p95:.1f} ms")
    if p95 > staging_cfg["max_p95_latency_ms"]:
        failures.append(f"p95 latency {p95:.1f} ms > {staging_cfg['max_p95_latency_ms']} ms")

    if failures:
        return GateDecision(False, failures)
    return GateDecision(True, [f"{len(sample)} held-out rows: " + ", ".join(facts)])
