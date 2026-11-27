# Convenience targets (need GNU make; on plain Windows copy the commands from docs/DEVELOPMENT.md)
PY = .venv/Scripts/python
.PHONY: setup bootstrap train promote serve mlflow-ui test drift

setup:
	python -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

# Train and run staged gates; Gate 2 requires an existing production baseline.
bootstrap: train promote

train:
	$(PY) -m usecases.churn.train

promote:
	$(PY) -m mlops.registry.promote churn --stage gate1
	$(PY) -m mlops.registry.promote churn --stage gate2
	$(PY) -m mlops.registry.promote churn --stage promote

serve:
	$(PY) -m uvicorn mlops.serving.app:app --port 8000

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	$(PY) -m pytest tests -q

drift:
	$(PY) -m scripts.replay_drift
