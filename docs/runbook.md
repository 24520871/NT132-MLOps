# Runbook

## M1 local MVP — verification record (2026-10-28)
Everything below was actually run on the developer machine (Windows 11, Python 3.14.3; mlflow 3.17.2,
scikit-learn 1.9.1, xgboost 3.4.1, fastapi 0.162.0, pandas 3.1.0).

- **Tests:** `pytest tests` -> 43 passed (data validation/cleaning, evaluation, quality gate, training + promotion
  against an isolated MLflow store, API contract incl. 422/503 and hot-reload of a new production version).
- **Real data:** 7,043 rows, 19 features + target, 0 nulls after cleaning, churn rate 26.54 %; 80/20 stratified split.
- **Training results (test split, threshold 0.5):**

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| logreg (selected) | 0.8415 | 0.5043 | 0.7834 | 0.6136 |
| xgboost | 0.8398 | 0.5169 | 0.7754 | 0.6203 |

- **Registry:** `churn` v1, gate `accepted` (thresholds AUC >= 0.80, recall >= 0.60), alias `production`.
- **Live API check:** uvicorn on port 8000 — `/ready` 200 (v1); high-risk sample -> churn p=0.90; low-risk sample ->
  no_churn p=0.015; `{"tenure": -1}` -> 422; `/docs` 200. Server stopped afterwards.
- **Not verified:** MLflow UI was not launched (only the registry/runs were queried through the client API).
  No Docker, Airflow, AWS or git yet.

## AWS (Stage F) — deploy the MVP on one EC2 instance
Region `ap-southeast-2`. Stack: `docker-compose.yml` + `docker-compose.prod.yml` (images from ECR, MLflow artifacts in S3).
Monitoring (Prometheus/Grafana) is added only after the smoke test below passes. Never put AWS keys in `.env`:
the containers use the EC2 instance role. Commands marked (laptop) run in PowerShell, (ec2) in the SSH session.
Status: written, **not yet run end to end**.

Prerequisites already created (steps 1-9 of the preparation): budget alert, S3 bucket (private, versioned),
3 ECR repos (`mlops-api`, `mlops-mlflow`, `mlops-trainer`), role `mlops-ec2-role` + instance profile,
key pair `mlops-key`, security group `mlops-sg` (22 and 8000 from the owner's IP only), EC2 t3.large with Docker.

### F1. Let containers reach the instance role (ec2 metadata hop limit) (laptop)
Containers are one network hop behind the instance. With the default hop limit of 1 they cannot read the role
credentials, and boto3 fails with `NoCredentialsError` when writing to S3.
```powershell
aws ec2 modify-instance-metadata-options --instance-id <INSTANCE_ID> --http-put-response-hop-limit 2 --http-tokens required --region ap-southeast-2
```

### F2. Build and push images (laptop, repo root, Docker Desktop running)
Copy `.env.example` to `.env` first and fill `POSTGRES_*`, `ECR_REGISTRY`, `AWS_REGION`, `S3_BUCKET` (uncomment the last three).
```powershell
$env:DOCKER_DEFAULT_PLATFORM = "linux/amd64"   # belt and braces; docker-compose.prod.yml also pins platform: linux/amd64
# After pushing, verify: docker image inspect <ECR>/mlops-api:latest --format "{{.Os}}/{{.Architecture}}"  -> linux/amd64
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.ap-southeast-2.amazonaws.com
docker compose -f docker-compose.yml -f docker-compose.prod.yml build
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile train build trainer
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile train push
```
Check in the ECR console that all three repos have a `latest` image.

### F3. DVC remote on S3 (laptop, venv active; needs the `dvc-s3` extra)
```powershell
.venv\Scripts\python -m pip install "dvc[s3]"
.venv\Scripts\dvc remote add -d s3remote s3://<BUCKET>/dvc
.venv\Scripts\dvc push
```
Commit the resulting `.dvc/config` change. The small public CSV is also in git, so the instance does not need `dvc pull` for the first run.

### F4. Start the stack (ec2)
```bash
git clone https://github.com/24520871/NT132-MLOps.git && cd NT132-MLOps
git checkout feat/aws-deploy            # or main once the PR is merged
cp .env.example .env && nano .env       # set POSTGRES_PASSWORD (your own), uncomment/fill ECR_REGISTRY, AWS_REGION, S3_BUCKET
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.ap-southeast-2.amazonaws.com
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-build
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```
Expected: postgres healthy, mlflow and api running (the api may report not ready until a model is promoted).

### F5. Train and promote on the instance (ec2)
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile train run --rm trainer
```
Check S3 has objects under `mlflow-artifacts/` (console or `aws s3 ls s3://<BUCKET>/mlflow-artifacts/ --recursive | head`).
The registry is a fresh Postgres DB, so `churn` v1 is created here; the laptop's `mlflow.db` is not carried over.

### F6. Smoke test (laptop)
```powershell
curl http://<EC2_PUBLIC_IP>:8000/ready
```
Then open `http://<EC2_PUBLIC_IP>:8000/docs` and call `POST /predict/churn` with the sample payload in `docs/DEVELOPMENT.md` ("Predict (PowerShell)", replace `127.0.0.1` with the instance IP). Expected: a churn probability.
MLflow UI through a tunnel (port 5000 is bound to localhost on the instance only):
```powershell
ssh -i $HOME\.ssh\mlops-key.pem -L 5000:localhost:5000 ubuntu@<EC2_PUBLIC_IP>
```
then browse `http://localhost:5000`.

### F7. Cost control and teardown
Stop when idle (disk and any Elastic IP are still billed; the public IP changes on the next start, so update the `8000` security-group rule only if *your* IP changed):
```powershell
aws ec2 stop-instances --instance-ids <INSTANCE_ID> --region ap-southeast-2
aws ec2 start-instances --instance-ids <INSTANCE_ID> --region ap-southeast-2
```
Full teardown (irreversible; empties the bucket and deletes images and the instance):
```powershell
aws ec2 terminate-instances --instance-ids <INSTANCE_ID> --region ap-southeast-2
aws ecr delete-repository --repository-name mlops-api --force --region ap-southeast-2
aws ecr delete-repository --repository-name mlops-mlflow --force --region ap-southeast-2
aws ecr delete-repository --repository-name mlops-trainer --force --region ap-southeast-2
aws s3 rb s3://<BUCKET> --force     # versioned bucket: also delete old versions in the console, or use a lifecycle rule
aws ec2 delete-security-group --group-id <SG_ID> --region ap-southeast-2   # after the instance is gone
aws ec2 delete-key-pair --key-name mlops-key --region ap-southeast-2
aws iam remove-role-from-instance-profile --instance-profile-name mlops-ec2-profile --role-name mlops-ec2-role
aws iam delete-instance-profile --instance-profile-name mlops-ec2-profile
aws iam delete-role-policy --role-name mlops-ec2-role --policy-name s3-access
aws iam detach-role-policy --role-name mlops-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam delete-role --role-name mlops-ec2-role
```
Teardown has not been tested yet; run it once after the smoke test and record the result here.

### F8. Monitoring on EC2 (Prometheus + Grafana overlay)
Run after F6 passes. Ports 3000/9090 are bound to `127.0.0.1` on the instance and are not in the security group:
reach them through an SSH tunnel only.
1. (laptop) Start the instance and read its new public IP (it changes on every start):
```powershell
aws ec2 start-instances --instance-ids i-045ce62867ca51c51 --region ap-southeast-2
aws ec2 wait instance-running --instance-ids i-045ce62867ca51c51 --region ap-southeast-2
aws ec2 describe-instances --instance-ids i-045ce62867ca51c51 --region ap-southeast-2 --query "Reservations[].Instances[].PublicIpAddress" --output text
```
If SSH times out, the home IP changed and the security group blocks it: add the new IP as `/32` for ports 22 and 8000
(`aws ec2 authorize-security-group-ingress --group-name mlops-sg --protocol tcp --port 22 --cidr <IP>/32 --region ap-southeast-2`).
Keeping both usual home IPs as two `/32` rules is fine; never open 22 to `0.0.0.0/0`.
2. (ec2) Get the latest code (history of the branch was rewritten once, so reset instead of pull):
```bash
cd ~/NT132-MLOps && git fetch origin && git reset --hard origin/feat/aws-deploy
```
3. (ec2) Add `GRAFANA_ADMIN_PASSWORD=<your own password>` to `.env` (never commit or paste it), then create the log folders:
```bash
mkdir -p reports/prediction_log reports/metrics && chmod -R 777 reports
```
Without write access the API silently fails to write the prediction log and those panels stay empty.
4. (ec2) ECR login (tokens last ~12 h), then start all three files:
```bash
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 454328770846.dkr.ecr.ap-southeast-2.amazonaws.com
docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d --no-build
docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.monitoring.yml ps
```
5. (laptop) Tunnel and browse `http://localhost:3000` (dashboard "MLOps - Churn overview") and `http://localhost:9090/targets`:
```powershell
ssh -i $HOME\.ssh\mlops-key.pem -L 3000:localhost:3000 -L 9090:localhost:9090 ubuntu@<EC2_PUBLIC_IP>
```
6. (laptop) Generate traffic so the rate panels show data (about 90 s, using `$body` from `docs/DEVELOPMENT.md`):
```powershell
1..90 | ForEach-Object { Invoke-RestMethod -Method Post -Uri http://<EC2_PUBLIC_IP>:8000/predict/churn -ContentType application/json -Body $body | Out-Null; Start-Sleep -Seconds 1 }
```
Verification record (2026-11-23): Prometheus targets `api`, `node`, `prometheus` all `up = 1`; Grafana showed "Serving model v1" and the
predicted-probability histogram after 9 successful predictions (probability 0.9033, model v1). **Not yet confirmed:** the request-rate,
p95-latency and predicted-churn panels under sustained traffic. "Data drift: no data" is expected until the drift/replay script has run.

## Airflow (Stage G) — local, docker compose overlay
Not deployed on EC2 yet (deliberately skipped when Stage G was written). Needs Docker Desktop running.
1. Add to `.env` (see `.env.example`): `AIRFLOW_DB_PASSWORD`, `AIRFLOW_ADMIN_PASSWORD`, `AIRFLOW_SECRET_KEY` (your own values).
2. Start: `docker compose -f docker-compose.yml -f docker-compose.airflow.yml up -d --build` (first build installs the ML stack into the image's `mlops-venv`, several minutes).
3. Open `http://localhost:8080` (user `admin`). DAGs are created paused: unpause `training_dag`, then "Trigger DAG".
   CLI alternative: `docker compose -f docker-compose.yml -f docker-compose.airflow.yml exec airflow-scheduler airflow dags trigger training_dag`.
4. Expected run: `validate_data` -> `train_evaluate` -> `gate1_to_staging` -> `gate2_to_production` -> `reload_api`, all green. In the MLflow UI the registered model `churn` gets a new version with `production`, the old one `previous_production`. A gate rejection shows the gate task and the ones after it as *skipped* (not failed) and production stays untouched; the reasons are tags on the model version (`gate_reasons`, `staging_gate_reasons`) and in the task log.
5. `ingest_dag` (daily, `@daily`) only validates the dataset.
Notes
- The DAG tasks call `$MLOPS_PYTHON` (separate venv in the Airflow image); the Airflow Python never imports scikit-learn/mlflow.
- Task logs: Airflow UI, or the `airflowlogs` volume. Airflow's metadata is in its own `airflow-db` Postgres (volume `airflowdb`).
- Memory: Airflow adds about 2 GB; on EC2 use >= t3.xlarge and add the Airflow services to a prod override (ECR image, S3 artifact root) first.
- Without Airflow the same flow still works: `docker compose run --rm trainer` (train + both gates).
- Not yet exercised end to end (see `docs/BUILD_PLAN.md` Stage G): first local run may need small fixes (volume permissions: set `AIRFLOW_UID` to the owner of `mlartifacts`).

### Known pitfalls
- SSH `Connection timed out`: home IP changed; see F8 step 1. An empty `$MYIP` gives `CIDR block /32 is malformed`.
- Compose files must be listed in order base, prod, monitoring (`-f ... -f ... -f ...`); `--no-build` because images come from ECR.
- `exec format error` in a container: image was built for ARM; rebuild with `DOCKER_DEFAULT_PLATFORM=linux/amd64`.
- `NoCredentialsError` / 403 on S3: hop limit not raised (F1), role missing S3 permissions, or wrong `S3_BUCKET`.
- API returns 503 on `/ready`: no model has the `production` alias yet (run F5), or the model was logged before the artifact root pointed at S3.
- `ports: !override` needs Docker Compose >= 2.24 (`docker compose version`).
