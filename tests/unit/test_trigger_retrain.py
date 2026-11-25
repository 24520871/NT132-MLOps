import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from mlops.monitoring.perf_monitor import evaluate_performance
from mlops.monitoring.trigger_retrain import MonitoringSignals, evaluate_retraining

CONFIG = {"monitoring": {"recall_drop_threshold": 0.05}}


def test_drift_above_configured_threshold_requests_retraining():
    decision = evaluate_retraining(
        MonitoringSignals(data_drift_detected=1.0, data_drift_simulated=0.0),
        CONFIG,
    )

    assert decision.should_retrain
    assert decision.triggering_conditions == ("observed_feature_drift",)
    assert decision.dag_run_id.startswith("stage_i_retrain__")
    assert "drift" in decision.reason


def test_performance_drop_with_available_ground_truth_requests_retraining():
    performance = evaluate_performance(
        pd.DataFrame(
            {
                "ground_truth": [1, 1, 1, 1, 0, 0],
                "prediction": [1, 1, 0, 0, 0, 0],
                "probability": [0.9, 0.8, 0.4, 0.3, 0.2, 0.1],
            }
        )
    )
    assert performance.available
    assert performance.recall == pytest.approx(0.5)

    decision = evaluate_retraining(
        MonitoringSignals(
            data_drift_detected=0.0,
            data_drift_simulated=0.0,
            current_recall=performance.recall,
            baseline_recall=0.7,
            production_version="12",
        ),
        CONFIG,
    )

    assert decision.should_retrain
    assert decision.triggering_conditions == ("observed_recall_drop",)


def test_missing_ground_truth_is_not_performance_degradation():
    performance = evaluate_performance(
        pd.DataFrame({"prediction": [1, 0], "probability": [0.8, 0.2]})
    )
    assert not performance.available

    decision = evaluate_retraining(
        MonitoringSignals(
            data_drift_detected=0.0,
            data_drift_simulated=0.0,
            current_recall=performance.recall,
            baseline_recall=0.9,
        ),
        CONFIG,
    )

    assert not decision.should_retrain
    assert not decision.triggering_conditions
    assert "unavailable" in decision.reason


def test_simulated_drift_and_performance_are_ignored():
    decision = evaluate_retraining(
        MonitoringSignals(
            data_drift_detected=1.0,
            data_drift_simulated=1.0,
            current_recall=0.2,
            baseline_recall=0.9,
        ),
        CONFIG,
    )

    assert not decision.should_retrain
    assert decision.reason == "Simulated metrics are ignored"
    assert not decision.triggering_conditions


def test_metrics_below_threshold_do_not_request_retraining():
    decision = evaluate_retraining(
        MonitoringSignals(
            data_drift_detected=0.0,
            data_drift_simulated=0.0,
            current_recall=0.82,
            baseline_recall=0.85,
        ),
        CONFIG,
    )

    assert not decision.should_retrain
    assert not decision.triggering_conditions


def test_multiple_conditions_are_reported():
    decision = evaluate_retraining(
        MonitoringSignals(
            data_drift_detected=1.0,
            data_drift_simulated=0.0,
            current_recall=0.7,
            baseline_recall=0.8,
            production_version="4",
        ),
        CONFIG,
    )

    assert decision.should_retrain
    assert decision.triggering_conditions == (
        "observed_feature_drift",
        "observed_recall_drop",
    )
    assert "; " in decision.reason


@pytest.mark.parametrize(
    ("signals", "reason_fragment"),
    [
        (MonitoringSignals(data_drift_detected=1.0), "simulation status is unavailable"),
        (
            MonitoringSignals(data_drift_detected=2.0, data_drift_simulated=0.0),
            "Invalid monitoring metric",
        ),
        (
            MonitoringSignals(data_drift_simulated=0.0, current_recall=1.2, baseline_recall=0.9),
            "Invalid monitoring metric",
        ),
        (
            MonitoringSignals(data_drift_simulated=0.0, current_recall=0.78, baseline_recall=0.8),
            "No retraining condition",
        ),
    ],
)
def test_missing_or_invalid_metric_inputs_fail_closed(signals, reason_fragment):
    decision = evaluate_retraining(signals, CONFIG)

    assert not decision.should_retrain
    assert reason_fragment in decision.reason
    assert not decision.triggering_conditions


def test_repeated_signal_for_same_production_version_has_same_dag_run_id():
    signals = MonitoringSignals(
        data_drift_detected=1.0,
        data_drift_simulated=0.0,
        production_version="8",
    )

    first = evaluate_retraining(signals, CONFIG)
    repeated = evaluate_retraining(signals, CONFIG)

    assert first.dag_run_id == repeated.dag_run_id
    assert first.dag_run_id != evaluate_retraining(
        MonitoringSignals(
            data_drift_detected=1.0,
            data_drift_simulated=0.0,
            production_version="9",
        ),
        CONFIG,
    ).dag_run_id


def test_monitoring_dag_imports_and_triggers_training_when_airflow_is_installed():
    if importlib.util.find_spec("airflow") is None:
        pytest.skip("Airflow is not installed in the local test environment")
    airflow_models = pytest.importorskip(
        "airflow.models", reason="Airflow package is present without the DAG model components"
    )
    DagBag = airflow_models.DagBag

    dag_path = Path(__file__).resolve().parents[2] / "airflow" / "dags"
    dagbag = DagBag(dag_folder=str(dag_path), include_examples=False)
    assert dagbag.import_errors == {}

    dag = dagbag.get_dag("monitoring_dag")
    assert dag is not None
    trigger = dag.get_task("trigger_training")
    assert trigger.trigger_dag_id == "training_dag"
    assert trigger.skip_when_already_exists
