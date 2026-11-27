"""Evaluate production metrics and trigger the existing training DAG when warranted."""
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


def _choose_retraining_path() -> str:
    """Route only a positive, valid decision from the separate ML environment."""
    context = get_current_context()
    task_instance = context["ti"]
    decision_output = task_instance.xcom_pull(task_ids="evaluate_metrics")
    if not isinstance(decision_output, str):
        raise ValueError("Retraining decision task did not return JSON")
    try:
        decision = json.loads(decision_output)
    except json.JSONDecodeError as exc:
        raise ValueError("Retraining decision task returned invalid JSON") from exc
    if not isinstance(decision, dict) or not isinstance(decision.get("should_retrain"), bool):
        raise ValueError("Retraining decision JSON is missing should_retrain")
    if decision["should_retrain"]:
        dag_run_id = decision.get("dag_run_id")
        if not isinstance(dag_run_id, str) or not dag_run_id:
            raise ValueError("Positive retraining decision is missing its deterministic dag_run_id")
        task_instance.xcom_push(key="dag_run_id", value=dag_run_id)
        return "trigger_training"
    return "no_retraining_needed"


with DAG(
    dag_id="monitoring_dag",
    description="Evaluate real production metrics and request retraining when configured conditions are met",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "monitoring"],
) as dag:
    evaluate_metrics = BashOperator(
        task_id="evaluate_metrics",
        bash_command="$MLOPS_PYTHON -m mlops.monitoring.trigger_retrain",
        do_xcom_push=True,
        execution_timeout=timedelta(minutes=2),
    )
    choose_path = BranchPythonOperator(
        task_id="choose_retraining_path",
        python_callable=_choose_retraining_path,
    )
    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training_dag",
        trigger_run_id="{{ ti.xcom_pull(task_ids='choose_retraining_path', key='dag_run_id') }}",
        conf={"usecase": "churn", "source": "stage_i_monitoring"},
        reset_dag_run=False,
        skip_when_already_exists=True,
        wait_for_completion=False,
    )
    no_retraining_needed = EmptyOperator(task_id="no_retraining_needed")

    evaluate_metrics >> choose_path >> [trigger_training, no_retraining_needed]
