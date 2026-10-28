# MLOps Platform — NT132 Group 4

University MLOps project. Source proposal: `D:\abcdefgh\NT\NT132\Proposal_Group4_NT132.docx` (Vietnamese).
Goal: end-to-end MLOps platform on AWS — data versioning, automated training/eval, model registry with promote/rollback, FastAPI serving, monitoring + drift detection, automated retraining. Use cases: Fraud Detection, Customer Churn, Recommendation, Demand Forecasting.

**At the start of every session, read `docs/BUILD_PLAN.md`** — it has the stage-by-stage build plan, decisions, working rules and open items. Update its checkboxes as work completes.

High-level plan: `C:\Users\pc\.claude\plans\d-abcdefgh-nt-nt132-proposal-group4-nt1-inherited-fox.md`

## Current scope: MVP only
Train churn models on a real public dataset (IBM Telco Customer Churn, `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`; fail loudly if missing, never fall back to synthetic), track in MLflow, quality-gate, promote via aliases, serve with FastAPI. Run locally, no Docker/Airflow/AWS yet. Do not build later layers until the MVP is approved and working.

UI decision: no extra UI framework (no Streamlit). Use FastAPI Swagger at `/docs` for predictions and the MLflow UI for runs/registry.

Full-project skeleton exists as stub files (docstring/comment only, each TODO names its stage). Layout:
- `src/mlops/{data,training,registry,serving,monitoring}/` shared platform logic (package is `mlops`, NOT `platform`, which shadows a stdlib module)
- `usecases/{churn,fraud,recsys,forecast}/` per-use-case schema/features/model/config (churn also has data.py, train.py)
- `airflow/dags/`, `docker/{airflow,serving,mlflow}/`, `docker-compose*.yml`, `infra/{terraform,prometheus,grafana}/`, `.github/workflows/`, `tests/{unit,integration}/`, `docs/`, `notebooks/`, `dvc.yaml`, `Makefile`, `pyproject.toml`, `.env.example`
Stages (stop for user OK after each; details in `docs/BUILD_PLAN.md`): A data -> B train/MLflow -> C gate/promote -> D FastAPI+tests [M1 local MVP] -> E Docker Compose -> F deploy MVP on AWS [M2] -> G Airflow -> H monitoring -> I auto-retrain+rollback [M3] -> J CI/CD -> K other use cases -> L final report [M4]. Stub "Stage X" labels use this lettering.

## Stack (full project)
Python, scikit-learn/XGBoost, MLflow, DVC, Airflow, FastAPI, Docker, GitHub Actions (+Trivy), Prometheus/Grafana, Evidently, PostgreSQL, AWS EC2/S3/ECR.

## Decisions
- One platform, many use cases: each in `usecases/<name>/` with the same interface (load data, features, train, evaluate); later DAGs/API are parameterized by use-case name.
- MLflow **aliases** (`candidate`, `staging`, `production`), not deprecated stages. Rollback = repoint `production` alias.
- MVP MLflow store: local SQLite (`sqlite:///mlflow.db`); Postgres + S3 later.
- Deploy target: `docker compose` on a single EC2 (>= t3.large), S3 for DVC/artifacts, ECR for images.
- Metrics: Precision, Recall, F1, AUC (prioritize recall/AUC for churn).
- Telco cleaning: `TotalCharges` is text with 11 blanks -> numeric, fill 0; drop `customerID`; `Churn` Yes/No -> 1/0; stratified 80/20 split, fixed seed.

## Commands (Windows, from project root; full list in README.md)
- Setup: `python -m venv .venv`, `.venv\Scripts\python -m pip install -r requirements.txt`, `.venv\Scripts\python -m pip install -e .`
- Train: `.venv\Scripts\python -m usecases.churn.train`; promote: `-m mlops.registry.promote churn`; serve: `-m uvicorn mlops.serving.app:app`; tests: `-m pytest tests -q`
- Set `MLFLOW_DISABLE_AGENT_HINT=1` to silence MLflow hint output.

## Rules
- Plan first; don't write code until the user approves.
- Ask before downloading anything (Kaggle needs a login; user supplies the CSV).
- Never commit `data/`, `.venv/`, `mlflow.db`, `mlruns/`.
- Don't present synthetic-data metrics as real results.
- Environment: Windows 11, Python 3.14, PowerShell. Not a git repo yet (ask before `git init`).

## Status
- [x] Scaffold: stub files for all stages, requirements, .gitignore, venv
- [x] A data, B train/MLflow, C gate/promote, D FastAPI + README — **M1 local MVP done 2026-10-28, 43 tests pass** (details: `docs/BUILD_PLAN.md` progress log)
- State: real Telco CSV in `data/raw/` (DVC-tracked, no git); `mlflow.db` has `churn` v1 = `production`; run `pip install -e .` in `.venv` first
- [ ] E Docker Compose, F AWS deploy (M2)
- [ ] G Airflow, H monitoring, I retrain+rollback (M3)
- [ ] J CI/CD, K other use cases, L report (M4)
