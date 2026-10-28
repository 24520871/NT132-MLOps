# Convenience targets (need GNU make; on plain Windows copy the commands from README.md)
PY = .venv/Scripts/python

setup:
	python -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

train:
	$(PY) -m usecases.churn.train

promote:
	$(PY) -m mlops.registry.promote churn

serve:
	$(PY) -m uvicorn mlops.serving.app:app --port 8000

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	$(PY) -m pytest tests -q
