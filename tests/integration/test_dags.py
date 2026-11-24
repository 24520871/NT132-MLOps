"""Airflow DAG files: always syntax-checked; fully imported when Airflow is installed (it is in the Airflow image)."""
from pathlib import Path

import pytest

DAG_DIR = Path(__file__).resolve().parents[2] / "airflow" / "dags"
DAG_FILES = sorted(DAG_DIR.glob("*.py"))


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
def test_dag_file_compiles(path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_dag_files_do_not_import_the_ml_stack():
    """DAGs run in Airflow's own Python; ML code must only be called through $MLOPS_PYTHON."""
    for path in DAG_FILES:
        text = path.read_text(encoding="utf-8")
        for banned in ("import mlflow", "import sklearn", "import xgboost", "from mlops", "import mlops"):
            assert banned not in text, f"{path.name} imports {banned!r}"


@pytest.fixture(scope="module")
def dagbag():
    pytest.importorskip("airflow")
    from airflow.models import DagBag

    return DagBag(dag_folder=str(DAG_DIR), include_examples=False)


def test_dags_import_without_errors(dagbag):
    assert dagbag.import_errors == {}
    assert {"ingest_dag", "training_dag"} <= set(dagbag.dags)


def test_training_dag_task_order_and_gate_skip_semantics(dagbag):
    dag = dagbag.get_dag("training_dag")
    order = ["validate_data", "train_evaluate", "gate1_to_staging", "gate2_to_production", "reload_api"]
    assert {t.task_id for t in dag.tasks} == set(order)
    for upstream, downstream in zip(order, order[1:]):
        assert downstream in dag.get_task(upstream).downstream_task_ids
    for gate in ("gate1_to_staging", "gate2_to_production"):  # rejection (exit 1) = skipped, not failed
        codes = dag.get_task(gate).skip_on_exit_code
        assert 1 in ([codes] if isinstance(codes, int) else codes)
