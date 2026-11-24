"""Daily data check: the dataset must exist and pass schema validation (same code path training uses).

Runs `mlops.data.check` in the separate `mlops-venv` ($MLOPS_PYTHON). A failure here means the data is
missing or invalid, before anyone starts a training run on it.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="ingest_dag",
    description="Validate the use case's raw data",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    params={"usecase": "churn"},
    default_args={"retries": 0, "execution_timeout": timedelta(minutes=10)},
    tags=["mlops", "data"],
) as dag:
    BashOperator(
        task_id="validate_data",
        bash_command="$MLOPS_PYTHON -m mlops.data.check {{ params.usecase }}",
    )
