# NT132 MLOps Platform — Group 4 (MVP: Telco churn)

Local MVP: validate and clean the IBM Telco churn data, train two models, track them in MLflow,
gate + promote the best one through registry aliases, and serve it with FastAPI.
(Docker, AWS, Airflow and monitoring come in later stages — see `docs/BUILD_PLAN.md`.)

## Requirements
Python 3.11+ (developed on 3.14, Windows 11). Commands below are for PowerShell from the project root.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

## Dataset
IBM Telco Customer Churn (Kaggle `blastchar/telco-customer-churn`, 7,043 rows, 26.5 % churn).
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`. If the file is missing, training stops with
a clear error — it never falls back to synthetic data. The file is tracked with DVC (`data/raw/*.csv.dvc`,
local cache only; no git repo or remote yet).

## Run the workflow
```powershell
# 1. Train both models (logistic regression + XGBoost); best by AUC is registered as `candidate`
.venv\Scripts\python -m usecases.churn.train

# 2. Quality gate + promote (exit code 0 = promoted, 1 = rejected, 2 = no candidate)
.venv\Scripts\python -m mlops.registry.promote churn

# 3. Serve (Swagger UI at http://127.0.0.1:8000/docs)
.venv\Scripts\python -m uvicorn mlops.serving.app:app --port 8000

# 4. Browse experiments, metrics and the model registry (http://127.0.0.1:5000)
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Predict (PowerShell):
```powershell
$body = @{gender="Female";SeniorCitizen=0;Partner="Yes";Dependents="No";tenure=5;PhoneService="Yes";
 MultipleLines="No";InternetService="Fiber optic";OnlineSecurity="No";OnlineBackup="No";
 DeviceProtection="No";TechSupport="No";StreamingTV="Yes";StreamingMovies="Yes";
 Contract="Month-to-month";PaperlessBilling="Yes";PaymentMethod="Electronic check";
 MonthlyCharges=89.5;TotalCharges=447.5} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict/churn -ContentType application/json -Body $body
```
Or use the "Try it out" button on `/docs`.

## API
| Endpoint | Purpose |
|---|---|
| `GET /health` | Process is up |
| `GET /ready` | 200 if a `production` model is loadable, else 503 with instructions |
| `POST /predict/churn` | Returns `prediction`, `label`, `probability`, `model_version`; 422 on invalid input, 503 if no production model |

## Configuration
`usecases/churn/config.yaml` holds the dataset path, split seed, model parameters, threshold and
quality-gate thresholds (`min_auc`, `min_recall`, max AUC drop vs production).
Env vars: `MLFLOW_TRACKING_URI` (default local `mlflow.db`), `MLFLOW_ARTIFACT_ROOT` (default `./mlruns`),
`DATA_PATH`, `PROJECT_ROOT`.

## Tests
```powershell
.venv\Scripts\python -m pytest tests -q
```

## Troubleshooting
- **`Dataset not found`** — put the CSV in `data/raw/` (see above).
- **API returns 503 `No 'production' model`** — run steps 1 and 2.
- **Promote says REJECTED** — the candidate failed the gate; the reasons are printed and stored as tags on the
  model version in MLflow. Production is left unchanged.
- **`ModuleNotFoundError: mlops`** — run `pip install -e .` inside the venv.
- **Start fresh** — stop MLflow/uvicorn, delete `mlflow.db` and `mlruns/`.

## Shutdown
Press Ctrl+C in the uvicorn / MLflow UI terminals.
