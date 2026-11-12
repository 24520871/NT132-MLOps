# Tổng quan dự án MLOps — NT132 Nhóm 4 (để chia task)

Cập nhật: 2026-11-12. Chi tiết từng bước: `docs/BUILD_PLAN.md`. Sơ đồ: `mlops_aws_architecture.md`.

## 1. Dự án là gì
Một nền tảng MLOps end-to-end trên AWS: quản lý phiên bản dữ liệu, huấn luyện và đánh giá tự động, model registry có promote/rollback, serving bằng FastAPI, giám sát và phát hiện drift, tự động retrain. Một platform dùng chung cho 4 use case: Churn (đang làm), Fraud, Recommendation, Demand Forecasting.

## 2. Luồng tổng thể

```
Developer → GitHub → GitHub Actions (test, build, Trivy) → ECR → EC2 (docker compose)

Dữ liệu (Kaggle/UCI) → DVC → S3 → Data Validation
   ↓
Airflow: Prep → Train → Evaluate → Gate 1 ──FAIL→ Reject
   ↓ PASS
MLflow Registry:  candidate → staging
   ↓
Gate 2 (kiểm tra staging tự động) ──FAIL→ giữ nguyên production
   ↓ PASS
production  (previous_production = bản cũ)
   ↓
FastAPI (Docker): /predict, /health, /ready  ──metrics──> Prometheus → Grafana
   ↓ predictions                                          Evidently (drift) → Alert
Vòng phản hồi: drift/giảm hiệu năng → Airflow retrain → Gate 1 + Gate 2 → promote/reject
   ↓
Cửa sổ theo dõi 24h sau promote → suy giảm → AUTO-ROLLBACK (production = previous_production)
```

### Vòng đời model (đã quyết định 2026-11-12)
- `candidate` → (Gate 1: metric offline) → `staging` → (Gate 2: kiểm tra tự động) → `production`. Không có bước duyệt thủ công.
- **Gate 1**: AUC ≥ 0.80, recall ≥ 0.60, không tệ hơn production về AUC.
- **Gate 2**: load model từ `staging`, kiểm tra schema/signature, dự đoán thử trên mẫu held-out (không lỗi, không NaN, tỉ lệ dương hợp lý), so sánh shadow với production, đo latency.
- **Auto-rollback**: trong 24h (hoặc N lát replay) sau promote, rollback nếu error rate hoặc p95 latency > 2 lần baseline, recall giảm > 0.05 so với lúc qua gate, hoặc prediction drift nặng. Hành động: `production` = `previous_production`, gắn tag `rolled_back=true` kèm lý do, chặn tự promote lại, gửi alert, API reload. An toàn: mỗi version rollback tối đa 1 lần, có cooldown, không có `previous_production` thì chỉ alert.
- API reload: router kiểm tra alias `production` mỗi 30–60 giây, thêm `POST /admin/reload` để gọi ngay.

> Lưu ý: phần staging, Gate 2 và auto-rollback mới là **thiết kế**, sẽ code ở Stage G và I. Code M1 hiện tại vẫn đi thẳng `candidate → production`.

## 3. Vai trò từng thành phần

| Thành phần | Vai trò | Vị trí trong repo | Trạng thái |
|---|---|---|---|
| Dữ liệu và validation | Nạp CSV Telco, kiểm tra schema/range/null, làm sạch, split 80/20 stratified. Thiếu file thì báo lỗi, không dùng dữ liệu giả. | `src/mlops/data/`, `usecases/churn/{schema,data}.py` | Xong |
| DVC | Version hóa dữ liệu (hiện local, chưa có remote S3) | `dvc.yaml`, `data/raw/*.dvc` | Local xong, S3 ở Stage F |
| Training | Logistic Regression và XGBoost dạng sklearn Pipeline, đánh giá, log MLflow | `src/mlops/training/`, `usecases/churn/{features,model,train}.py` | Xong |
| Quality gate (Gate 1) | Hàm thuần quyết định có cho lên bước tiếp theo không | `src/mlops/training/quality_gate.py` | Xong |
| MLflow | Tracking run và registry (SQLite, sẽ chuyển Postgres + S3) | `mlflow.db` (không commit) | Xong (local) |
| Promote | Đọc metric candidate, chạy gate, gán alias | `src/mlops/registry/promote.py` | Xong (candidate → production), cần thêm staging |
| Rollback | Trỏ lại `production` về `previous_production` | `src/mlops/registry/rollback.py` | Stub (Stage I) |
| Serving | FastAPI: `/predict/churn`, `/health`, `/ready`; 503 khi chưa có model, 422 khi request sai | `src/mlops/serving/` | Xong, thiếu auto-reload |
| Gate 2 | Kiểm tra model ở staging | `src/mlops/training/staging_check.py` (mới) | Chưa làm (Stage G) |
| Airflow | Điều phối pipeline train/retrain | `airflow/dags/` | Chưa làm (Stage G) |
| Monitoring | Prometheus, Grafana, Evidently, alert | `src/mlops/monitoring/`, `infra/` | Chưa làm (Stage H) |
| Auto-retrain và auto-rollback | Phát hiện drift → retrain; theo dõi sau promote → rollback | `monitoring/trigger_retrain.py`, `registry/rollback.py` | Chưa làm (Stage I) |
| Docker / Compose | Đóng gói Postgres, MLflow, API, trainer | `docker/`, `docker-compose*.yml` | Chưa làm (Stage E) |
| AWS | EC2, S3, ECR, IAM | `infra/` | Chưa làm (Stage F) |
| CI/CD | Test, build, Trivy, deploy | `.github/workflows/` | Chưa làm (Stage J) |
| Use case khác | Fraud, Recsys, Forecast, chỉ cần implement interface use case | `usecases/{fraud,recsys,forecast}/` | Chưa làm (Stage K) |

## 4. Kết quả M1 (local MVP, 43 test pass)
Dữ liệu thật 7,043 dòng, churn 26.5%. Trên test split: Logistic Regression AUC 0.8415 / recall 0.7834; XGBoost AUC 0.8398 / recall 0.7754. Logreg được đăng ký là `churn` v1 và đang là `production`.

## 5. Đề xuất chia task (4 người)

| Người | Nhiệm vụ | Phụ thuộc |
|---|---|---|
| A | **E → F**: Docker Compose, deploy AWS (S3, ECR, IAM, EC2), runbook, teardown. Sau đó phần CD của J. | Làm đầu tiên, B và C cần E |
| B | **G → I**: Airflow DAG, chuyển promote sang `candidate → staging → production`, Gate 2, API auto-reload, `rollback.py`, auto-rollback, trigger retrain. | E |
| C | **H**: metrics Prometheus, Grafana, Evidently, expose tín hiệu cho auto-rollback. Sau đó phần CI của J. | E; cấp tín hiệu cho I |
| D | **K** (3 use case: Fraud, Forecast, Recsys) rồi **L** (demo, load test, báo cáo, slide). | Độc lập, bắt đầu ngay được |

Ghi chú: B và C cần thống nhất sớm tên metric Prometheus và ngưỡng rollback (B dùng, C tạo ra). D cần chọn dataset trước khi bắt đầu.

## 6. Cần chốt trước khi bắt đầu
- Xác nhận ngưỡng Gate 1 (AUC ≥ 0.80, recall ≥ 0.60 đang là mặc định do Claude chọn). Ngưỡng Gate 2 và rollback ở mục 2 là đề xuất.
- `git init` và tạo GitHub repo chung (hiện repo chỉ ở máy local; cần cho cả nhóm và Stage J).
- Tài khoản AWS (credits sinh viên?), region, cảnh báo ngân sách, key pair EC2.
- Dataset cho Fraud (Kaggle credit-card fraud), Forecast (Rossmann hoặc store-item demand), Recsys (MovieLens).
- Mỗi lần chỉ làm một stage, xong thì dừng chờ duyệt (quy tắc trong `CLAUDE.md`).
