"""Airflow DAG import and pre-training validation dependency checks."""
from pathlib import Path

import pytest


def test_training_dag_validation_blocks_downstream_tasks():
    try:
        from airflow.models import DagBag
    except (ImportError, ModuleNotFoundError):
        pytest.skip("full Airflow installation is required for DAG import validation")

    dag_folder = Path(__file__).resolve().parents[2] / "airflow" / "dags"
    dag_bag = DagBag(dag_folder=str(dag_folder), include_examples=False)

    assert not dag_bag.import_errors
    dag = dag_bag.get_dag("training_dag")
    assert dag is not None
    validate = dag.get_task("validate_data")
    train = dag.get_task("train_evaluate")
    model_gate = dag.get_task("gate1_to_staging")

    assert "train_evaluate" in validate.downstream_task_ids
    assert train.trigger_rule == "all_success"
    assert "gate1_to_staging" in train.downstream_task_ids
    assert model_gate.trigger_rule == "all_success"
