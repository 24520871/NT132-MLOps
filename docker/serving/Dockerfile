# Shared image for the API and the trainer (same libs => models load identically).
FROM python:3.12-slim

# libgomp1: OpenMP runtime needed by xgboost on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY usecases ./usecases
RUN pip install --no-cache-dir -e .

ENV PROJECT_ROOT=/app \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONUNBUFFERED=1

# /mlartifacts is a shared volume; must be writable by the non-root user
RUN useradd --create-home appuser && mkdir /mlartifacts && chown appuser /mlartifacts
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
