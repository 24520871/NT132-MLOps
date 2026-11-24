# Stage G — Airflow orchestration: giải thích

Tài liệu này giải thích Stage G làm gì, vì sao làm, và các phần ghép với nhau thế nào. Chi tiết lệnh chạy nằm ở `docs/runbook.md` (mục "Airflow (Stage G)"), kế hoạch và checkbox ở `docs/BUILD_PLAN.md`.

> **Trạng thái (2026-11-25):** code, test và tài liệu đã viết xong, nhưng **chưa chạy được**: máy chưa có `.venv` nên chưa chạy `pytest`, và Docker Desktop đang tắt nên chưa build/chạy Airflow. Mới có `py_compile` và `docker compose config` đạt. Chưa redeploy EC2 (chủ động bỏ qua).

---

## 1. Stage G giải quyết vấn đề gì?

Trước Stage G, muốn có một model mới trong production thì người vận hành phải tự gõ lệnh theo thứ tự:

```
train  ->  promote  ->  (hy vọng API tự nhận model mới)
```

Hai vấn đề:

1. **Thủ công, không có lịch sử chạy.** Không ai biết lần train gần nhất chạy lúc nào, bước nào lỗi, vì sao model bị từ chối.
2. **Quy trình quá ngắn.** Model đạt Gate 1 (chỉ nhìn metric offline AUC/recall) là lên production ngay, không có bước kiểm tra "model này thật sự chạy ổn khi được load và gọi không".

Stage G làm ba việc:

| # | Việc | Kết quả |
|---|---|---|
| 1 | Đưa pipeline vào **Airflow DAG** | Một nút bấm chạy cả pipeline, có UI, log từng bước, retry, lịch sử |
| 2 | Thêm **staging + Gate 2** | Model đi `candidate -> staging -> production`, có bước kiểm tra tự động thứ hai |
| 3 | **API tự nhận model mới** | Không cần restart API sau khi promote |

Stage G **không** làm: tự động retrain khi drift, rollback tự động (đó là Stage I), DAG monitoring, deploy lên EC2.

---

## 2. Vòng đời model (alias trong MLflow Registry)

```
 train
   |
   v
 candidate --[Gate 1: metric offline]--> staging --[Gate 2: kiểm tra tự động]--> production
                |                                        |                          (cũ -> previous_production)
                v                                        v
        từ chối: gắn tag, bỏ alias               từ chối: gắn tag, bỏ alias
        production giữ nguyên                    production giữ nguyên
```

- Alias thay cho "stage" cũ của MLflow (đã deprecated).
- Mỗi lần quyết định (đạt hoặc không) alias `candidate`/`staging` đều bị xoá, nên không có model "treo".
- Lý do bị từ chối được ghi thành **tag** trên model version: `gate`, `gate_reasons`, `staging_gate`, `staging_gate_reasons`. Xem trong MLflow UI.

### Gate 1 (đã có từ Stage C)
Nhìn metric đo lúc train: `auc >= 0.80`, `recall >= 0.60`, và AUC không tệ hơn production hiện tại. Ngưỡng ở mục `gate` trong `usecases/churn/config.yaml`.

### Gate 2 (mới, `src/mlops/training/staging_check.py`)
Load model từ `staging` và thử nó trên một mẫu dữ liệu held-out (tối đa 500 dòng của tập test, lấy cố định theo seed):

| Kiểm tra | Ý nghĩa | Ngưỡng (`staging_gate`) |
|---|---|---|
| Signature | Cột đầu vào của model khớp cột feature | phải khớp |
| Output hợp lệ | Không NaN/inf, xác suất nằm trong [0, 1] | bắt buộc |
| Tỉ lệ dự đoán churn | Model không dự đoán "toàn churn" hay "toàn không churn" | 0.05 – 0.70 |
| So với production | Trung bình \|p_staging − p_production\| trên cùng các dòng | <= 0.25 |
| Độ trễ | p95 của 20 lần dự đoán từng dòng | <= 500 ms |

Gate 2 báo **tất cả** lỗi một lượt (không dừng ở lỗi đầu tiên). Các ngưỡng này do tôi đề xuất, chưa được nhóm xác nhận.

---

## 3. Các thành phần

```
                      Airflow (container)
   +----------------------------------------------------------+
   |  webserver (UI :8080)     scheduler (LocalExecutor)      |
   |                                                          |
   |  training_dag:                                           |
   |   validate_data -> train_evaluate -> gate1_to_staging    |
   |        -> gate2_to_production -> reload_api              |
   |                                                          |
   |  mỗi task = BashOperator gọi  $MLOPS_PYTHON -m ...       |
   |                       |                                  |
   |                       v                                  |
   |            mlops-venv (Python riêng, có sklearn/xgboost/ |
   |            mlflow)  -- tách khỏi Python của Airflow      |
   +-----------------------|----------------------------------+
                           | HTTP                    ^ metadata
                           v                         |
   mlflow server <---- registry/aliases        airflow-db (Postgres riêng)
        |
        v
   api (FastAPI)  <---- POST /admin/reload  (task cuối của DAG)
```

### 3.1. Hai DAG (`airflow/dags/`)

**`training_dag`** (chạy tay, `schedule=None`; Stage I sẽ tự kích hoạt khi có drift):

| Task | Lệnh thực chạy | Làm gì |
|---|---|---|
| `validate_data` | `python -m mlops.data.check churn` | Kiểm tra CSV tồn tại, đúng schema (cùng code path với train) |
| `train_evaluate` | `python -m usecases.churn.train` | Train logreg + XGBoost, log MLflow, đăng ký model tốt nhất làm `candidate` |
| `gate1_to_staging` | `python -m mlops.registry.promote churn --stage gate1` | Gate 1 -> alias `staging` |
| `gate2_to_production` | `python -m mlops.registry.promote churn --stage gate2` | Gate 2 -> alias `production` |
| `reload_api` | `python -m mlops.serving.reload_client` | Báo API đọc lại alias `production` (retry 2 lần) |

Điểm cần nhớ: **model bị gate từ chối không làm DAG "fail"**. CLI thoát với mã 1, task được đặt `skip_on_exit_code=1` nên hiện là *skipped* (cùng các task phía sau), production giữ nguyên. Mã 2 (không có gì để promote) và mọi lỗi khác vẫn là *failed*.

DAG nhận tham số `usecase` (mặc định `churn`), nên sau này đổi use case chỉ cần truyền `--conf '{"usecase": "..."}'`.

**`ingest_dag`**: chạy `@daily`, chỉ gồm `validate_data`. Báo sớm nếu dữ liệu mất hoặc sai schema, trước khi ai đó bắt đầu train.

### 3.2. Vì sao dùng venv riêng thay vì cài thẳng vào Airflow?
Airflow ghim rất nhiều phiên bản thư viện; scikit-learn/xgboost/pandas/mlflow của dự án cũng ghim phiên bản riêng. Cài chung dễ xung đột. Nên image Airflow có `/opt/airflow/mlops-venv` chứa stack ML, DAG chỉ gọi `$MLOPS_PYTHON`. File DAG không import `mlops`/`mlflow`/`sklearn` (test `test_dag_files_do_not_import_the_ml_stack` kiểm tra điều này).

### 3.3. Docker (`docker/airflow/Dockerfile`, `docker-compose.airflow.yml`)
- Image: `apache/airflow:2.10.5-python3.12` + `mlops-venv`.
- Overlay chạy thêm 4 service: `airflow-db` (Postgres riêng, **không** đụng volume Postgres của MLflow), `airflow-init` (chạy một lần: migrate DB, tạo user `admin`), `airflow-webserver` (chỉ mở `127.0.0.1:8080`), `airflow-scheduler`.
- Cùng volume `mlartifacts` với MLflow/API (để ghi artifact), mount `./data` chỉ đọc (để train và Gate 2 đọc CSV).
- Mật khẩu/secret lấy từ `.env`, không có trong repo.
- Là overlay riêng (giống `docker-compose.monitoring.yml`) nên `docker compose up` bình thường không bị ảnh hưởng.

### 3.4. API tự nhận model mới (`src/mlops/serving/`)
Trước đây router hỏi MLflow alias `production` ở **mọi request**. Giờ:
- Tin kết quả tra alias trong `MODEL_POLL_SECONDS` giây (mặc định 30), giảm tải cho MLflow.
- `POST /admin/reload` ép tra lại ngay. DAG gọi endpoint này ở task cuối nên model mới có hiệu lực tức thì.
- Nếu đặt `ADMIN_TOKEN`, endpoint đòi header `X-Admin-Token` (nên đặt khi cổng API truy cập được từ ngoài).
- Nếu quên reload, polling vẫn bắt kịp sau tối đa 30 giây.

---

## 4. Danh sách file thay đổi

| File | Việc |
|---|---|
| `airflow/dags/training_dag.py`, `ingest_dag.py` | DAG mới |
| `docker/airflow/Dockerfile`, `docker-compose.airflow.yml` | Image và overlay Airflow |
| `src/mlops/registry/promote.py` | `stage_candidate` (Gate 1) + `promote_staging` (Gate 2), CLI `--stage` |
| `src/mlops/training/staging_check.py` | Logic Gate 2 (hàm thuần, không phụ thuộc MLflow) |
| `src/mlops/data/check.py` | CLI kiểm tra dữ liệu |
| `src/mlops/serving/router.py`, `app.py`, `reload_client.py` | Polling, `/admin/reload`, client gọi reload |
| `usecases/churn/config.yaml` | Mục `staging_gate` |
| `docker-compose.yml`, `.env.example` | `ADMIN_TOKEN`, biến Airflow |
| `tests/unit/test_staging_check.py` (mới); `tests/integration/test_training.py`, `test_api.py`, `test_dags.py`, `tests/conftest.py` | Test |
| `docs/BUILD_PLAN.md`, `docs/runbook.md`, `docs/DEVELOPMENT.md`, `CLAUDE.md` | Cập nhật tài liệu |

Lệnh cũ vẫn dùng được: `python -m mlops.registry.promote churn` (không tham số) chạy cả hai gate, nên service `trainer` và `make promote` vẫn đi tới `production`.

---

## 5. Cách chạy (khi đã có Docker Desktop)

```powershell
# .env: thêm AIRFLOW_DB_PASSWORD, AIRFLOW_ADMIN_PASSWORD, AIRFLOW_SECRET_KEY (giá trị của bạn)
docker compose -f docker-compose.yml -f docker-compose.airflow.yml up -d --build
# Mở http://localhost:8080 (admin), bật (unpause) training_dag, bấm Trigger
```

Kết quả mong đợi: 5 task xanh; trong MLflow UI model `churn` có version mới mang alias `production`, version cũ mang `previous_production`; `GET /ready` của API trả version mới.

Thử đường từ chối: hạ thấp ngưỡng trong `config.yaml` (ví dụ `min_auc: 0.999`) rồi trigger lại, task `gate1_to_staging` sẽ *skipped*, production không đổi, lý do nằm trong tag `gate_reasons`.

---

## 6. Chưa được kiểm chứng và việc còn lại

- Chưa chạy `pytest tests -q` (cần tạo `.venv`).
- Hai chỗ dựa vào API thư viện mà chưa thử thật: `mlflow.models.get_model_info` đọc signature, và thuộc tính `skip_on_exit_code` của BashOperator.
- Image Airflow chưa build, DAG chưa chạy. Rủi ro đã biết: quyền ghi volume `mlartifacts` (xử lý bằng `AIRFLOW_UID`, xem runbook).
- Chưa có override production cho Airflow (ảnh ECR, artifact S3), và Airflow cần thêm khoảng 2 GB RAM, nên dùng EC2 >= t3.xlarge. Redeploy EC2 được bỏ qua theo yêu cầu.
- Ngưỡng Gate 2 là đề xuất, cần nhóm xác nhận.
- Để Stage I làm tiếp: tự trigger `training_dag` khi drift, rollback tự động, `monitoring_dag`.
