# Build Plan — NT132 MLOps Platform (Group 4)

Read this file and `CLAUDE.md` at the start of every session. Update the status checkboxes here as work completes.
Roadmap: **A–D local MVP -> E–F containerized MVP deployed on AWS -> G–I Airflow/monitoring/auto-retrain -> J–L CI/CD, other use cases, final report.** Current target: Stage A. (Stub files' "Stage X" TODO labels follow this lettering.)

## Working rules (every stage)
1. Do one stage at a time. After finishing, run the stage's tests/commands, report real results (never claim success unrun), update the checkboxes below and `CLAUDE.md` status, then **stop and wait for the user's OK**.
2. Ask before: downloading anything, deleting `mlflow.db` or other existing files, `git init`, installing heavy new tooling.
3. No secrets in files; config via env vars + `config.yaml`. Never commit `data/`, `.venv/`, `mlflow.db`, `mlruns/`, `.env`.
4. Match the stub layout. Package is `mlops` (not `platform`). Keep code small; no abstractions beyond the use-case interface (schema / load+clean / features / model / train / evaluate).

## Decisions (change only if the user says so)
- Dataset: IBM Telco Customer Churn CSV at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 rows, 21 cols, 26.5% churn). Missing file -> raise a clear error; **never** fall back to synthetic data.
- Models: logistic regression baseline + XGBoost, as sklearn Pipelines (preprocessing inside the saved model).
- Metrics: ROC-AUC, precision, recall, F1 (+ confusion matrix artifact). Prioritize recall/AUC.
- Gate defaults (`usecases/churn/config.yaml`): `min_auc: 0.80`, `min_recall: 0.60`, and the candidate must not be worse than the current production model on AUC. (User to confirm.)
- MLflow: local SQLite `sqlite:///mlflow.db`; registry aliases `candidate` / `production` (+ `previous_production` for rollback). No MLflow stages.
- UI: none extra. FastAPI Swagger `/docs` + MLflow UI.
- Packaging: `pyproject.toml` with `pip install -e .` so `mlops` and `usecases` import cleanly; pytest config there too.
- Python: local 3.14 for MVP; Docker later may use 3.11 (check version-specific pins then).

---

## Stage A — Data (real dataset, validation, cleaning)
Goal: a loader that returns a clean, validated Telco DataFrame or fails loudly.

Tasks
- [x] `pyproject.toml`: project metadata, setuptools packages (`mlops`, `usecases`), pytest config. Add `pytest`, `pyyaml`, `dvc` to `requirements.txt`; `pip install -e .`.
- [x] `src/mlops/config.py`: load `usecases/<name>/config.yaml` + env overrides (MLflow URI, data path).
- [x] `usecases/churn/schema.py`: expected 21 columns, dtypes, allowed categorical values, numeric ranges (tenure 0–72, charges >= 0).
- [x] `src/mlops/data/validation.py`: generic `validate(df, schema)` -> raises with a readable list of problems (missing columns, bad dtypes, unexpected categories, out-of-range, null rates).
- [x] `src/mlops/data/ingest.py`: resolve path, raise `FileNotFoundError` with instructions (Kaggle `blastchar/telco-customer-churn`) if missing.
- [x] `usecases/churn/data.py`: **rewrite** — remove the synthetic generator. `load_data()` = ingest -> clean (`TotalCharges` to numeric, blanks -> 0; drop `customerID`; `Churn` Yes/No -> 1/0) -> validate. Add `split()` = stratified 80/20, fixed seed (split happens before any fitting).
- [x] DVC (local only): `dvc init`, `dvc add data/raw/...csv` (needs `git init` approval first; otherwise `dvc init --no-scm`). Fill `dvc.yaml`/`.dvcignore`.
- [x] `tests/unit/test_validation.py`: small in-memory fixtures (not the real CSV) — good frame passes; missing column, bad category, negative charge, blank TotalCharges handling.
Done when: real CSV loads -> 7,043 rows, 20 feature cols + target, no nulls, `TotalCharges` numeric, churn rate ≈ 26.5%; tests pass; removing the CSV gives the clear error.

## Stage B — Train + MLflow
Goal: one command trains two models, logs everything, registers the best as `candidate`.

Tasks
- [x] `usecases/churn/features.py`: `ColumnTransformer` (numeric: impute+scale; categorical: one-hot, handle_unknown=ignore).
- [x] `usecases/churn/model.py`: logistic regression (`class_weight=balanced`) and XGBoost (`scale_pos_weight`) pipelines; params from `config.yaml`.
- [x] `src/mlops/training/evaluate.py`: precision/recall/F1/AUC + confusion matrix; threshold 0.5 default (configurable).
- [x] `src/mlops/training/runner.py`: generic loop — load, split, fit each model, evaluate, log params/metrics/artifacts/model (with input example + signature) to MLflow; pick best by AUC.
- [x] `usecases/churn/train.py`: **rewrite** thin entry: `python -m usecases.churn.train`; registers best model as `churn` and sets alias `candidate`.
- [x] Old synthetic `mlflow.db`/`mlruns` deleted (user chose option 1).
- [x] `Makefile` targets (setup, train, promote, serve, mlflow-ui, test); README has the PowerShell commands.
- [x] Tests: evaluate() on known arrays; train smoke test on a tiny fixture frame (temp MLflow URI).
Done when: MLflow UI shows two runs with metrics (expect AUC ≈ 0.83–0.85), best model registered with `candidate` alias; no leakage (fit only on train split); tests pass.

## Stage C — Quality gate + promotion
Goal: promotion is explicit and never silent.

Tasks
- [x] `src/mlops/training/quality_gate.py`: pure function `evaluate_gate(candidate_metrics, thresholds, production_metrics=None) -> Decision(accepted, reasons)`.
- [x] `src/mlops/registry/promote.py`: read candidate metrics from its MLflow run; run gate; if accepted set `previous_production` = old production, `production` = candidate; if rejected leave production untouched, set tag `gate=rejected` + reasons. CLI: `python -m mlops.registry.promote churn`.
- [x] Thresholds read from `config.yaml`.
- [x] `tests/unit/test_quality_gate.py`: passes, fails AUC, fails recall, worse than production, no production yet (first model allowed if thresholds met).
Done when: running promote with a good candidate sets `production`; with thresholds set impossibly high it rejects, logs the reason, and production alias is unchanged. (`rollback.py` stays a stub until Stage G.)

## Stage D — FastAPI serving + tests
Goal: predictions from the production model via API, with correct errors.

Tasks
- [x] `src/mlops/serving/schemas.py`: pydantic request (Telco features, `Literal` categories, ranges) and response (`prediction`, `probability`, `model_version`).
- [x] `src/mlops/serving/router.py`: lazily load `models:/churn@production`; cache; reload method.
- [x] `src/mlops/serving/app.py`: `POST /predict/churn`, `GET /health` (process up), `GET /ready` (production model loadable). No production model -> HTTP 503 with an actionable message ("train and promote first"). Invalid body -> 422.
- [x] `tests/integration/test_api.py` (FastAPI `TestClient`): valid request, invalid request (422), no-production (503), health/ready. Use a tiny model fixture + temp MLflow store.
- [x] Add `fastapi`, `uvicorn`, `httpx` to `requirements.txt`; Makefile `serve` target (`uvicorn mlops.serving.app:app`).
Done when: with a promoted model, `/docs` Swagger returns a probability for a sample customer; with no model, `/ready` and `/predict/churn` return 503; tests pass.

## MVP wrap-up
- [x] README.md: install, get dataset, train, promote, serve, predict (curl + Swagger), MLflow UI, troubleshooting, shutdown.
- [x] Full test run; record results and any gaps in `docs/runbook.md`.
- [x] Update `CLAUDE.md` status; list remaining work.

---

## Stage E — Containerize the MVP (local Docker Compose)
Goal: the MVP runs identically in containers on the laptop. Prereq: Docker Desktop installed (check `docker --version` first and report).

Tasks
- [ ] `docker/serving/Dockerfile` (FastAPI, Python 3.11-slim, non-root user) and `docker/mlflow/Dockerfile` (MLflow server).
- [ ] `docker-compose.yml`: `postgres` (MLflow backend store, healthcheck), `mlflow` (backend = Postgres, artifacts = shared volume for now), `api` (serving), one-off `trainer` service (`docker compose run --rm trainer`). Config via `.env` (`.env.example` committed, `.env` ignored; no hardcoded credentials).
- [ ] Switch MLflow URI from SQLite to the containerized server via env var; retrain (old SQLite runs are not migrated).
- [ ] Pin dependency versions (known-good set from the working local MVP) and verify they install on Python 3.11.
- [ ] Verify: `docker compose config` valid -> `up -d` -> healthchecks green -> train in container -> promote -> `curl /predict/churn` works -> `docker compose down`.
Done when: the full train -> promote -> predict flow works with only Docker commands.

## Stage F — Deploy the MVP on AWS  (milestone M2: "MVP running in the cloud")
Goal: the Stage E stack runs on one EC2 instance, with S3 for data/artifacts and ECR for images. Every paid or account-changing action needs the user's explicit approval; the user runs `aws configure` / enters credentials themselves — never ask for or handle keys in chat.

What I need from the user first
- AWS account (student credits?), chosen region, a billing budget alert set, a CLI profile or SSO configured, an EC2 key pair, their public IP for security-group rules.

Tasks
- [ ] Write `docs/runbook.md` AWS section: exact CLI commands for each resource, plus teardown commands. Terraform (`infra/terraform/`) is optional later; do manual/CLI first so it is understood and debuggable.
- [ ] S3 bucket (versioned, private): DVC remote (`data/`) + MLflow artifact root.
- [ ] ECR repos (`mlops-api`, `mlops-mlflow`, `mlops-trainer`); build, tag, push images from the laptop.
- [ ] IAM: instance role (S3 bucket access + ECR pull, least privilege) — no access keys on the instance.
- [ ] EC2: t3.large (>= 8 GB RAM), Ubuntu LTS, Docker + compose plugin; security group: SSH only from the user's IP, API/MLflow ports only from the user's IP (or API public if user wants a public demo; MLflow/Airflow/Grafana never public).
- [ ] `docker-compose.prod.yml`: ECR image refs, S3 artifact root, restart policies, env from instance (no secrets in repo).
- [ ] `dvc remote add` S3 + `dvc push`; MLflow artifacts to S3.
- [ ] Deploy, run training on the instance, promote, smoke-test `/ready` and `/predict/churn` from the laptop.
- [ ] Cost controls: stop instance when idle, billing alarm, documented teardown.
Done when: a prediction request from the laptop to the EC2 API returns a probability from a model whose artifacts live in S3; teardown commands documented and tested.

## Stage G — Airflow orchestration
Goal: training runs as an Airflow DAG instead of a manual command.
- [ ] `docker/airflow/Dockerfile`, Airflow in compose (pick version, LocalExecutor, Postgres metadata DB). Likely needs t3.xlarge on EC2 — confirm RAM before deploying.
- [ ] `airflow/dags/ingest_dag.py` (validate data) and `training_dag.py`: ingest/validate -> train -> evaluate -> quality gate -> promote/reject. Run the training step in the separate trainer container (DockerOperator/BashOperator), not Airflow's own Python env, to avoid dependency clashes.
- [ ] `tests/integration/test_dags.py` (DAGs import cleanly); trigger a run, verify the MLflow run and registry change.
- [ ] Redeploy to EC2 and re-verify.

## Stage H — Monitoring
- [ ] API metrics (`serving/metrics.py`): request rate, latency, errors; prediction logging (`prediction_log.py`).
- [ ] Prometheus (+ node-exporter, cAdvisor) and Grafana with provisioned dashboards and an alert rule (`infra/prometheus`, `infra/grafana`).
- [ ] Evidently data/prediction drift report (`monitoring/drift.py`), run by `monitoring_dag.py`; drift score exposed as a metric; HTML report saved to S3.
- [ ] Verify with normal vs. artificially shifted data.

## Stage I — Automated retraining + rollback
- [ ] `monitoring/trigger_retrain.py`: drift/performance decay above threshold -> trigger `training_dag` -> gate vs. production -> promote or reject.
- [ ] `registry/rollback.py`: repoint `production` to `previous_production`; the API picks it up via reload.
- [ ] Simulate delayed ground truth by replaying held-out data in time slices; demo script: drift injected -> retrain -> promote -> serve, plus a rollback.

## Stage J — CI/CD (needs `git init` + a GitHub repo)
- [ ] `ci.yml`: ruff, pytest, DAG import check, `docker build`, Trivy (fail on HIGH/CRITICAL).
- [ ] `cd.yml`: build + push to ECR, deploy to EC2 (SSM or SSH). Prefer GitHub OIDC -> IAM role over long-lived keys; the user adds any repo secrets themselves.

## Stage K — Remaining use cases
- [ ] Fraud (e.g. Kaggle credit-card fraud; PR-AUC/recall), Forecast (e.g. Rossmann / store-item demand; MAE/RMSE), Recsys (e.g. MovieLens; precision@k). Each only implements the use-case interface (+ its own request schema); no platform changes should be needed — if they are, fix the abstraction. The user supplies/approves each dataset.

## Stage L — Final evaluation and report
- [ ] End-to-end demo script, API load test (latency), architecture diagram, finalize `docs/architecture.md`, `docs/runbook.md`, report/slides material. Tear down or stop AWS resources.

## Milestones
| Milestone | Stages | Result |
|---|---|---|
| M1 Local MVP | A–D | Telco churn trained, gated, promoted, served locally |
| M2 Cloud MVP | E–F | Same MVP in Docker on AWS (EC2 + S3 + ECR) |
| M3 Automation | G–I | Airflow pipeline, monitoring/drift, auto-retrain, rollback |
| M4 Complete | J–L | CI/CD, all four use cases, final demo and report |

## Open items needing the user
- [x] Telco CSV placed in `data/raw/`
- [ ] Confirm gate thresholds (default AUC >= 0.80, recall >= 0.60)
- [ ] `git init` — user said NOT YET (DVC runs with `--no-scm`)
- [x] Old synthetic `mlflow.db` deleted

## Progress log
- **M1 local MVP (Stages A–D) completed 2026-10-28**, run unattended at the user's request. 43 tests pass. Real Telco data: 7,043 rows, churn 26.5 %. Results on the 20 % test split: logreg AUC 0.8415 / recall 0.7834; XGBoost AUC 0.8398 / recall 0.7754. Best (logreg) registered as `churn` v1 and promoted to `production` (gate defaults AUC>=0.80, recall>=0.60 — chosen by Claude, user has not confirmed).
- Deviations / notes: MLflow 3.17's default skops serialization rejected our pipelines (untrusted numpy types), so models are logged with `cloudpickle` (acceptable for our own trusted store; revisit if models ever come from untrusted sources). DVC works without git (`dvc init --no-scm`). Next up: Stage E (Docker Compose) — needs Docker Desktop check first; waiting for the user's OK.
