"""Decide whether observed production metrics warrant an Airflow retraining run."""
import hashlib
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlencode

import httpx

from mlops.config import load_config, tracking_uri

log = logging.getLogger(__name__)

DEFAULT_PROMETHEUS_URL = "http://prometheus:9090"
METRIC_NAMES = (
    "data_drift_detected",
    "data_drift_simulated",
    "model_recall",
    "model_version_info",
)


@dataclass(frozen=True)
class MonitoringSignals:
    """Metric values used by the decision function; absent values remain unavailable."""

    data_drift_detected: float | None = None
    data_drift_simulated: float | None = None
    current_recall: float | None = None
    baseline_recall: float | None = None
    production_version: str | None = None


@dataclass(frozen=True)
class RetrainingDecision:
    should_retrain: bool
    reason: str
    metric_values: dict[str, float | str | None]
    triggering_conditions: tuple[str, ...]
    dag_run_id: str | None = None


class MonitoringMetricsError(RuntimeError):
    """Prometheus or registry metrics could not be read safely."""


def _valid_unit_metric(value: float | None) -> bool:
    return value is None or (
        isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0
    )


def evaluate_retraining(
    signals: MonitoringSignals,
    config: Mapping[str, object],
) -> RetrainingDecision:
    """Evaluate real drift/performance signals without importing or invoking Airflow.

    Drift's configured threshold is applied by ``detect_drift`` before it exports
    ``data_drift_detected``. Performance uses the documented recall-drop threshold
    only when both current and gate-time recall are available.
    """
    values = {
        "data_drift_detected": signals.data_drift_detected,
        "data_drift_simulated": signals.data_drift_simulated,
        "current_recall": signals.current_recall,
        "baseline_recall": signals.baseline_recall,
        "production_version": signals.production_version,
    }
    numeric_values = {
        name: value for name, value in values.items() if name != "production_version"
    }
    invalid = [
        name
        for name, value in numeric_values.items()
        if value is not None
        and (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
        )
    ]
    invalid.extend(
        name
        for name, value in (
            ("current_recall", signals.current_recall),
            ("baseline_recall", signals.baseline_recall),
        )
        if not _valid_unit_metric(value)
    )
    if signals.data_drift_detected not in (None, 0.0, 1.0):
        invalid.append("data_drift_detected")
    if signals.data_drift_simulated not in (None, 0.0, 1.0):
        invalid.append("data_drift_simulated")
    if signals.production_version is not None and not isinstance(signals.production_version, str):
        invalid.append("production_version")
    if invalid:
        return RetrainingDecision(
            False,
            "Invalid monitoring metric values: " + ", ".join(sorted(set(invalid))),
            values,
            (),
        )

    if signals.data_drift_simulated != 0.0:
        reason = (
            "Simulated metrics are ignored"
            if signals.data_drift_simulated == 1.0
            else "Monitoring metrics are insufficient: simulation status is unavailable"
        )
        return RetrainingDecision(False, reason, values, ())

    monitoring_config = config.get("monitoring", {})
    if not isinstance(monitoring_config, Mapping):
        return RetrainingDecision(False, "Monitoring configuration is invalid", values, ())
    recall_drop_threshold = monitoring_config.get("recall_drop_threshold")
    if recall_drop_threshold is not None and (
        not isinstance(recall_drop_threshold, (int, float))
        or not 0.0 <= float(recall_drop_threshold) <= 1.0
    ):
        return RetrainingDecision(False, "Configured recall_drop_threshold is invalid", values, ())

    conditions: list[str] = []
    descriptions: list[str] = []
    if signals.data_drift_detected == 1.0:
        conditions.append("observed_feature_drift")
        descriptions.append("observed feature drift crossed its configured detection threshold")

    if (
        isinstance(recall_drop_threshold, (int, float))
        and signals.current_recall is not None
        and signals.baseline_recall is not None
        and signals.baseline_recall - signals.current_recall > recall_drop_threshold
    ):
        conditions.append("observed_recall_drop")
        descriptions.append(
            "observed recall dropped by "
            f"{signals.baseline_recall - signals.current_recall:.4f}, exceeding "
            f"the configured {float(recall_drop_threshold):.4f} limit"
        )

    if conditions:
        stable_key = json.dumps(
            {"production_version": signals.production_version, "conditions": conditions},
            sort_keys=True,
            separators=(",", ":"),
        )
        dag_run_id = (
            "stage_i_retrain__"
            + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:24]
        )
        return RetrainingDecision(
            True,
            "; ".join(descriptions),
            values,
            tuple(conditions),
            dag_run_id,
        )

    if signals.current_recall is None or signals.baseline_recall is None:
        reason = "No retraining condition met; labeled performance metrics are unavailable"
    else:
        reason = "No retraining condition met"
    return RetrainingDecision(False, reason, values, ())


def _query_prometheus(base_url: str, metric: str, timeout: float = 10.0) -> list[dict[str, object]]:
    query_url = f"{base_url.rstrip('/')}/api/v1/query?{urlencode({'query': metric})}"
    try:
        response = httpx.get(query_url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise MonitoringMetricsError(
            f"Could not query Prometheus metric {metric}: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise MonitoringMetricsError(f"Prometheus returned an invalid response for metric {metric}")
    data = payload.get("data")
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, list):
        raise MonitoringMetricsError(f"Prometheus response is missing results for metric {metric}")
    return [sample for sample in result if isinstance(sample, dict)]


def _first_value(samples: list[dict[str, object]]) -> float | None:
    if not samples:
        return None
    value = samples[0].get("value")
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        parsed = float(value[1])
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _production_version(samples: list[dict[str, object]]) -> str | None:
    if not samples:
        return None
    metric = samples[0].get("metric")
    version = metric.get("version") if isinstance(metric, dict) else None
    return version if isinstance(version, str) and version else None


def _baseline_recall(
    model_version_samples: list[dict[str, object]], model_name: str
) -> float | None:
    if not model_version_samples:
        return None
    metric = model_version_samples[0].get("metric")
    version = metric.get("version") if isinstance(metric, dict) else None
    if not isinstance(version, str) or not version:
        return None

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri())
        model_version = client.get_model_version(model_name, version)
        run = client.get_run(model_version.run_id)
    except Exception as exc:
        raise MonitoringMetricsError(
            f"Could not read gate-time recall for production v{version}: {type(exc).__name__}"
        ) from exc
    recall = run.data.metrics.get("recall")
    return float(recall) if recall is not None else None


def collect_monitoring_signals(
    *,
    config: Mapping[str, object],
    prometheus_url: str | None = None,
) -> MonitoringSignals:
    """Read current Prometheus gauges and gate-time recall for the served version."""
    base_url = prometheus_url or os.getenv("PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL)
    metrics = {name: _query_prometheus(base_url, name) for name in METRIC_NAMES}
    simulated = _first_value(metrics["data_drift_simulated"])
    current_recall = _first_value(metrics["model_recall"]) if simulated == 0.0 else None
    return MonitoringSignals(
        data_drift_detected=_first_value(metrics["data_drift_detected"]),
        data_drift_simulated=simulated,
        current_recall=current_recall,
        baseline_recall=(
            _baseline_recall(metrics["model_version_info"], str(config["model_name"]))
            if current_recall is not None
            else None
        ),
        production_version=_production_version(metrics["model_version_info"]),
    )


def main() -> None:
    """CLI used by the Airflow monitoring DAG; emits one JSON decision for XCom."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = load_config("churn")
        signals = collect_monitoring_signals(config=config)
        decision = evaluate_retraining(signals, config)
    except (MonitoringMetricsError, KeyError, OSError) as exc:
        log.error("Retraining decision could not be evaluated: %s", exc)
        sys.exit(2)
    print(
        json.dumps(
            {
                "should_retrain": decision.should_retrain,
                "reason": decision.reason,
                "metric_values": decision.metric_values,
                "triggering_conditions": decision.triggering_conditions,
                "dag_run_id": decision.dag_run_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
