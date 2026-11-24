# Airflow 2.10 (webserver + scheduler, LocalExecutor). The ML code lives in a separate venv
# (/opt/airflow/mlops-venv) so Airflow's own pinned dependencies never clash with the project's.
# DAG tasks call that interpreter through $MLOPS_PYTHON; the DAG files themselves import nothing from `mlops`.
FROM apache/airflow:2.10.5-python3.12

USER root
# libgomp1: OpenMP runtime needed by xgboost on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /app && chown airflow:0 /app

USER airflow
WORKDIR /app
COPY --chown=airflow:0 requirements.txt .
RUN python -m venv /opt/airflow/mlops-venv \
    && /opt/airflow/mlops-venv/bin/pip install --no-cache-dir -r requirements.txt

COPY --chown=airflow:0 pyproject.toml .
COPY --chown=airflow:0 src ./src
COPY --chown=airflow:0 usecases ./usecases
RUN /opt/airflow/mlops-venv/bin/pip install --no-cache-dir -e .

ENV PROJECT_ROOT=/app \
    MLOPS_PYTHON=/opt/airflow/mlops-venv/bin/python \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONUNBUFFERED=1
