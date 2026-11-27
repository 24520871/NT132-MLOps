"""Training pipeline: validate -> train -> Gate 1 -> Gate 2 -> promote -> API reload.

Every step runs the platform's own CLI inside the separate `mlops-venv` of the Airflow image ($MLOPS_PYTHON),
so Airflow's dependencies never clash with scikit-learn / xgboost / mlflow. Nothing from `mlops` is imported here.

A gate rejection is a normal outcome, not a failure: the promote CLI exits 1, the task is marked *skipped*
(skip_on_exit_code=1) and so are the tasks after it; production stays untouched. Exit 2 (nothing to promote)
and any crash fail the task.

Trigger manually, optionally for another use case:  airflow dags trigger training_dag --conf '{"usecase": "churn"}'
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

REJECTED = 1  # a rejected gate is skipped, and all downstream tasks require all_success

with DAG(
    dag_id="training_dag",
    description="validate -> train -> Gate 1 (staging) -> Gate 2 -> production promotion",
    schedule=None,  # manual for now; Stage I triggers it from drift / performance decay
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    params={"usecase": "churn"},
    default_args={"retries": 0, "execution_timeout": timedelta(minutes=30)},
    tags=["mlops", "training"],
) as dag:
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="$MLOPS_PYTHON -m mlops.data.check {{ params.usecase }}",
    )
    # Trains every model, logs runs to MLflow, registers the best by AUC as `candidate`.
    train = BashOperator(
        task_id="train_evaluate",
        bash_command="$MLOPS_PYTHON -m usecases.{{ params.usecase }}.train",
    )
    gate1_staging = BashOperator(
        task_id="gate1_to_staging",
        bash_command="$MLOPS_PYTHON -m mlops.registry.promote {{ params.usecase }} --stage gate1",
        skip_on_exit_code=REJECTED,
    )
    gate2_production = BashOperator(
        task_id="gate2_check",
        bash_command="$MLOPS_PYTHON -m mlops.registry.promote {{ params.usecase }} --stage gate2",
        skip_on_exit_code=REJECTED,
    )
    promote_production = BashOperator(
        task_id="promote_production",
        bash_command="$MLOPS_PYTHON -m mlops.registry.promote {{ params.usecase }} --stage promote",
        skip_on_exit_code=REJECTED,
    )
    reload_api = BashOperator(
        task_id="reload_api",
        bash_command="$MLOPS_PYTHON -m mlops.serving.reload_client",
        retries=2,
        retry_delay=timedelta(seconds=30),
    )

    validate_data >> train >> gate1_staging >> gate2_production >> promote_production >> reload_api
