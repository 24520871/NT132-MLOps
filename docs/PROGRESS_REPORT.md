# Progress Report — NT132 MLOps Platform, Group 4

Date: 2026-10-28 · Milestone reached: **M1 local MVP (Stages A–D)** · Source: `Proposal_Group4_NT132.docx`

## 1. Summary
A working local MVP for **Customer Churn** is built and verified: real IBM Telco data is validated and cleaned,
two models are trained and tracked in MLflow, a configurable quality gate decides promotion through registry
aliases, and a FastAPI service serves the production model. 43 automated tests pass. Cloud deployment, Airflow,
monitoring/drift, automated retraining, CI/CD and the other three use cases are **not started**.

## 2. What was done
| Step | Work | Evidence |
|---|---|---|
| Planning | Read the proposal, designed the architecture, full-project skeleton and a staged build plan (`docs/BUILD_PLAN.md`); `CLAUDE.md` keeps context for future sessions | Files exist in repo |
| A. Data | Loader for the real Telco CSV (fails loudly if missing, no synthetic fallback), cleaning (`TotalCharges` blanks -> 0, drop ID, Yes/No -> 1/0), schema validation, stratified 80/20 split with fixed seed, DVC tracking (local) | 7,043 rows, 0 nulls, churn 26.54 %; 10 validation tests |
| B. Training | sklearn Pipelines (preprocessing inside the model), logistic regression + XGBoost, precision/recall/F1/AUC + confusion matrix, everything logged to MLflow with the dataset SHA-256; best model by AUC registered as `candidate` | 2 MLflow runs; see results below |
| C. Gate + promotion | Pure-logic quality gate (min AUC, min recall, not worse than production); promote command that never promotes silently, records reasons as tags, keeps `previous_production` | Accept and reject paths tested; real v1 promoted |
| D. Serving | FastAPI: `/health`, `/ready`, `POST /predict/churn`; 422 on invalid input, 503 with instructions when no production model; model router reloads automatically when the production alias moves | 16 API tests + live run with curl |
| Docs | README (install/run/troubleshoot), runbook with verification record | `README.md`, `docs/runbook.md` |

Test-split results (threshold 0.5): logistic regression AUC 0.8415 / recall 0.7834 / precision 0.5043 / F1 0.6136 (selected);
XGBoost AUC 0.8398 / recall 0.7754 / precision 0.5169 / F1 0.6203. Registered as `churn` v1, gate accepted, alias `production`.

## 3. Checklist against the proposal
Legend: ✅ done · 🟡 partial · ⬜ not started

### A. Development & CI/CD layer
- ⬜ GitHub repository (no `git init` yet, by the team's choice)
- ⬜ GitHub Actions: tests, Docker build, Trivy scan, deploy (stub files only)
- ⬜ Docker images / AWS ECR (Dockerfiles are empty stubs)

### B. Data layer
- 🟡 Dataset management: Telco churn ✅; fraud, recommendation, demand forecasting ⬜
- 🟡 DVC: dataset tracked locally (`.csv.dvc`, `--no-scm`) ✅; no S3 remote ⬜
- ⬜ AWS S3 storage (data, model artifacts)
- ✅ Data validation: schema, missing values, dtypes, value ranges, row count, allowed categories

### C. Orchestration & training layer
- ⬜ Apache Airflow (DAG files are stubs)
- ✅ Data preparation: cleaning + imputation/scaling/one-hot encoding (feature engineering is minimal)
- 🟡 Model training: churn ✅ (logreg + XGBoost); other use cases ⬜
- 🟡 Model evaluation: Precision, Recall, F1, AUC ✅; MAE/RMSE (forecasting) ⬜
- ✅ Quality gate: thresholds + comparison with the production model

### D. Model management layer
- ✅ MLflow tracking: params, metrics, confusion matrix, model artifact, data hash
- 🟡 Model registry: `candidate` and `production` aliases ✅ (aliases instead of deprecated stages); `staging` alias not used
- ✅ Promote / reject based on the gate (run as an explicit command; not yet triggered automatically)
- 🟡 Rollback: `previous_production` alias is recorded on promotion; `rollback.py` itself not written

### E. Model serving layer
- ✅ FastAPI prediction API with input validation and error handling
- ✅ Model router: serves the version tagged `production`, hot-reloads on change
- ⬜ Docker packaging of the API
- ⬜ AWS EC2 deployment

### F. Monitoring & feedback layer
- ⬜ Prometheus metrics (CPU, RAM, request rate, latency, errors)
- ⬜ Grafana dashboards and alerts
- ⬜ Evidently data / prediction drift
- ⬜ Automated retraining on drift

### Technologies (proposal section 4)
✅ Python · ✅ scikit-learn / XGBoost · ✅ MLflow · 🟡 DVC · ✅ FastAPI · ✅ pytest (extra) ·
⬜ Airflow · ⬜ Docker · ⬜ GitHub Actions · ⬜ Prometheus/Grafana · ⬜ Evidently ·
⬜ PostgreSQL (MVP uses SQLite for MLflow) · ⬜ AWS EC2 / S3 / ECR

### Expected results (proposal section 5)
- 🟡 Automated pipeline data -> train -> evaluate -> manage -> deploy: works for churn via 3 commands; not orchestrated or deployed
- 🟡 Reproducible pipeline with data/model versioning: dataset hash + local DVC + MLflow versions ✅; remote storage ⬜
- 🟡 REST API with FastAPI and Docker on AWS: FastAPI ✅; Docker/AWS ⬜
- ⬜ Monitoring with drift and performance-decay detection
- ⬜ Automatic retraining and model selection
- 🟡 Tested on 4 use cases: churn only (1 of 4)

Count for the 24 layer items (A–F): 7 fully done, 6 partial, 11 not started. The MVP deliberately covered only the
data, training, registry and serving parts.

## 4. Decisions and deviations
- Used MLflow **aliases** (`candidate`/`production`/`previous_production`) because stages are deprecated; the proposal's Candidate/Staging/Production maps onto these.
- No Streamlit/UI framework: Swagger (`/docs`) and the MLflow UI act as the interface (team decision, matches the proposal's tech list).
- Models are logged with `cloudpickle` because MLflow's default skops format rejected the pipelines; acceptable for a trusted private store.
- Gate thresholds (AUC >= 0.80, recall >= 0.60) were chosen by the assistant and **not yet confirmed** by the team.
- Package is named `mlops` (not `platform`) to avoid shadowing a Python standard-library module.

## 5. Limitations and risks
- MLflow UI was not launched during verification (runs/registry were queried via the client API).
- Metrics come from one fixed split; no cross-validation, tuning or threshold optimisation yet.
- Precision is about 0.50 (many false alarms) because the model favours recall; acceptable for churn but should be discussed.
- Python 3.14 locally; Docker images will target 3.11, so dependency versions must be re-verified in Stage E.
- Single machine, no CI, no git history: work is not yet backed up or versioned in a repository.

## 6. Next steps
1. **Stage E** – Docker Compose (Postgres, MLflow, API, trainer); check Docker Desktop first.
2. **Stage F** – deploy the MVP to AWS (EC2, S3, ECR, IAM); needs the team's AWS account, budget alert and approval for every paid action.
3. **Stages G–I** – Airflow DAGs, Prometheus/Grafana, Evidently drift, auto-retrain, rollback script.
4. **Stages J–L** – GitHub Actions + Trivy, fraud/forecast/recommendation use cases, final demo and report.
5. Team decisions needed: confirm gate thresholds, when to `git init`, AWS account/region.
