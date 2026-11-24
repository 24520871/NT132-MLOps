# Hướng dẫn phát triển — NT132 MLOps Platform (MVP: Telco churn)

Local MVP: validate and clean the IBM Telco churn data, train two models, track them in MLflow,
gate + promote the best one through registry aliases, and serve it with FastAPI.
(Docker, AWS, Airflow and monitoring come in later stages — see `docs/BUILD_PLAN.md`.)

## Requirements
Python 3.11+ (developed on 3.14, Windows 11). Commands below are for PowerShell from the project root.

## Quick start (new teammate)
```powershell
git clone https://github.com/24520871/NT132-MLOps.git
cd NT132-MLOps
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest tests -q        # all tests should pass
make bootstrap                                  # train + promote -> a `production` model (or run steps 1-2 below)
.venv\Scripts\python -m uvicorn mlops.serving.app:app --port 8000   # Swagger at /docs
```
`mlflow.db` and `mlruns/` are local and git-ignored: every teammate trains their own copy.
Do not commit `.env`, credentials, `.pem`/`.key` files or Terraform state.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

## Dataset
IBM Telco Customer Churn (Kaggle `blastchar/telco-customer-churn`, 7,043 rows, 26.5 % churn).
The CSV is public and small (~1 MB), so it is committed at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
and works right after `git clone`. If the file is missing, training stops with a clear error — it never falls
back to synthetic data. `data/raw/*.csv.dvc` records its DVC hash; there is no DVC remote yet (S3 comes in
Stage F, after which larger datasets go through DVC instead of git). Only the Telco CSV is un-ignored;
anything else under `data/` stays out of git.

## Run the workflow
```powershell
# 1. Train both models (logistic regression + XGBoost); best by AUC is registered as `candidate`
.venv\Scripts\python -m usecases.churn.train

# 2. Gate 1 (metrics) -> `staging`, then Gate 2 (staging check on held-out rows) -> `production`
#    exit code 0 = promoted, 1 = rejected by a gate, 2 = nothing to promote
#    (--stage gate1 or --stage gate2 runs just one step)
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
| `POST /admin/reload` | Re-check the `production` alias now (otherwise the API re-checks every `MODEL_POLL_SECONDS`, default 30). Needs header `X-Admin-Token` only if `ADMIN_TOKEN` is set |
| `POST /predict/churn` | Returns `prediction`, `label`, `probability`, `model_version`; 422 on invalid input, 503 if no production model |

## Configuration
`usecases/churn/config.yaml` holds the dataset path, split seed, model parameters, threshold and
Gate 1 thresholds (`gate`: `min_auc`, `min_recall`, max AUC drop vs production) and Gate 2 thresholds
(`staging_gate`: sample size, positive-rate range, max mean |dp| vs production, max p95 latency).
Env vars: `MLFLOW_TRACKING_URI` (default local `mlflow.db`), `MLFLOW_ARTIFACT_ROOT` (default `./mlruns`),
`DATA_PATH`, `PROJECT_ROOT`, `MODEL_POLL_SECONDS`, `ADMIN_TOKEN`.

## Tests
```powershell
.venv\Scripts\python -m pytest tests -q
```

## Drift replay and monitoring
Run the deterministic synthetic replay (no project dataset or API calls are used):
```powershell
.venv\Scripts\python -m scripts.replay_drift --seed 42
# Or, when GNU Make is available:
make drift
```
The replay writes simulated drift metrics to `reports/metrics/drift.prom` and HTML summaries to
`reports/drift/`. Prometheus reads the textfile through node-exporter. Recall/AUC are exported only
when the prediction log contains valid ground-truth labels; the serving log does not include labels by
default, so those metrics are normally unavailable.

## Troubleshooting
- **`Dataset not found`** — put the CSV in `data/raw/` (see above).
- **API returns 503 `No 'production' model`** — run steps 1 and 2.
- **Promote says REJECTED** — the candidate failed the gate; the reasons are printed and stored as tags on the
  model version in MLflow. Production is left unchanged.
- **`ModuleNotFoundError: mlops`** — run `pip install -e .` inside the venv.
- **Start fresh** — stop MLflow/uvicorn, delete `mlflow.db` and `mlruns/`.

## Shutdown
Press Ctrl+C in the uvicorn / MLflow UI terminals.
