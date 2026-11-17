# Chia task — mục tiêu: có dashboard (Stage E → H, chưa AWS, chưa Airflow)

Quyết định 2026-11-15. Chi tiết stage: `docs/BUILD_PLAN.md`. Mục tiêu: `docker compose up` rồi mở Grafana thấy dashboard với dữ liệu thật từ API đang chạy.

Không làm trong đợt này: F (AWS), G (Airflow), J (CI/CD), K (use case khác), staging / Gate 2 / rollback (để G và I). Số liệu drift là dữ liệu replay mô phỏng, phải ghi rõ khi báo cáo.

## Phân công

| Người | Phần | File sở hữu |
|---|---|---|
| **Bạn (Stage E, tích hợp)** | Dockerfile API / MLflow / trainer, `docker-compose.yml` (Postgres, MLflow, API, trainer), MLflow chuyển sang server, ghim `requirements.txt` (Python 3.11), `.env.example`, gộp code và chạy thử cả stack | `docker/`, `docker-compose.yml`, `requirements.txt`, `.env.example` |
| **Teammate 1 (metrics và Grafana)** | Metrics Prometheus trong API, ghi log dự đoán, cấu hình Prometheus + alert, Grafana provisioning và dashboard JSON | `src/mlops/serving/metrics.py`, `prediction_log.py`, `infra/prometheus/`, `infra/grafana/`, `docker-compose.monitoring.yml` |
| **Teammate 2 (drift và replay)** | Báo cáo drift Evidently, kiểm tra hiệu năng, script replay dữ liệu bình thường / bị lệch, xuất drift score thành metric | `src/mlops/monitoring/`, `scripts/`, `Makefile` (target `drift`) |

## Hợp đồng metric (chốt trước khi viết code)

| Metric | Loại | Do ai tạo |
|---|---|---|
| `api_requests_total{endpoint,status}` | counter | Teammate 1 |
| `api_request_duration_seconds` | histogram | Teammate 1 |
| `churn_predictions_total{label}` | counter | Teammate 1 |
| `churn_probability` | histogram | Teammate 1 |
| `model_version_info{model,version}` | gauge | Teammate 1 |
| `data_drift_share` | gauge | Teammate 2 |
| `data_drift_detected` (0/1) | gauge | Teammate 2 |
| `model_recall`, `model_auc` (từ replay có nhãn, mô phỏng) | gauge | Teammate 2 |

Cách đưa drift vào Prometheus: Teammate 1 và 2 thống nhất MỘT cách ngay từ đầu (đề xuất: script drift ghi vào file textfile, hoặc một endpoint `/metrics` nhỏ riêng). Ghi lựa chọn vào đầu file này khi chốt.

**Đã chốt (Teammate 1, 2026-11-16): textfile collector.** Script drift ghi `reports/metrics/drift.prom` (định dạng Prometheus, tên metric đúng bảng trên, không label; dùng `prometheus_client.write_to_textfile` để ghi nguyên tử). Thư mục này được mount vào node-exporter; Prometheus scrape node-exporter nên không cần exporter riêng. Log dự đoán: `reports/prediction_log/predictions.jsonl` (đọc bằng `mlops.serving.prediction_log.read_prediction_log`), chỉ ghi khi API có env `PREDICTION_LOG_PATH` (overlay monitoring đã đặt sẵn).

## Checklist

### Bạn — Stage E (xong 2026-11-16, đã merge PR #1)
- [x] Kiểm tra `docker --version` và Docker Desktop chạy được.
- [x] `docker/serving/Dockerfile` (Python 3.12-slim, user không phải root), `docker/mlflow/Dockerfile`.
- [x] `docker-compose.yml`: `postgres` (healthcheck), `mlflow`, `api`, `trainer` (chạy một lần). Mật khẩu và cấu hình đọc từ `.env`.
- [x] Đổi `MLFLOW_TRACKING_URI` sang server (qua biến môi trường), train lại.
- [x] Ghim version thư viện, cài thử trên Python 3.12 (xgboost 3.4 cần >= 3.12; Evidently chưa kiểm tra, việc của Teammate 2).
- [x] Kiểm tra: `docker compose up -d` rồi train rồi promote rồi gọi `/predict/churn` rồi `down`.
- [x] Đẩy khung compose và Dockerfile API sớm (ngày đầu) để hai bạn kia dùng được.

### Teammate 1 — Stage H (metrics và dashboard)
- [x] `serving/metrics.py` với các metric ở bảng trên; gắn `/metrics` vào `app.py`; test không làm hỏng test API cũ.
- [x] `serving/prediction_log.py`: ghi request và dự đoán (file hoặc bảng), cho Teammate 2 dùng.
- [x] `infra/prometheus/prometheus.yml` (scrape API, và exporter drift nếu có), `alerts.yml` (lỗi cao, latency cao, drift).
- [x] `infra/grafana/`: provisioning datasource và dashboard JSON. Panel: request/s, p95 latency, tỉ lệ lỗi, tỉ lệ dự đoán churn, phân phối xác suất, version model đang chạy, drift score, kết quả gate gần nhất.
- [x] `docker-compose.monitoring.yml` (Prometheus, Grafana, tùy chọn node-exporter / cAdvisor); mật khẩu Grafana lấy từ `.env`.

### Teammate 2 — Stage H (drift và replay)
- [ ] Kiểm tra Evidently trên Python 3.11 và 3.14. Đã thử Python 3.12 với pandas 3.1.0 của project: Evidently 0.7.23 cài được nhưng lỗi import `infer_dtype_from_object`; dùng SciPy hiện có làm phương án thay thế, không đổi dependency.
- [x] `monitoring/drift.py`: so feature tham chiếu/current bằng KS và chi-square; loại target/metadata; xuất `data_drift_share`, `data_drift_detected` và HTML report vào `reports/`.
- [x] `monitoring/perf_monitor.py`: tính recall / AUC chỉ khi prediction log có ground-truth hợp lệ; nếu không thì để unavailable.
- [x] `scripts/replay_drift.py`: tạo normal và shifted synthetic feature batches trong bộ nhớ; đánh dấu kết quả simulated.
- [x] `make drift`; unit tests cho drift/performance/replay và Prometheus output.

### Cả nhóm — kiểm thử cuối
- [ ] `docker compose up` rồi dashboard có dữ liệu.
- [ ] Replay dữ liệu lệch làm `data_drift_detected` = 1 và dashboard phản ánh.
- [x] Toàn bộ test suite cũ và mới pass (77 tests).
- [ ] Không có bí mật trong repo (`.env` chỉ ở máy từng người).

## Quy ước làm việc
- Mỗi người một nhánh: `feat/docker-base`, `feat/monitoring-dashboard`, `feat/drift-replay`. Merge vào `main` bằng pull request, `git pull` trước khi bắt đầu.
- Chỉ người sở hữu mới sửa file của mình (bảng trên). Cần sửa file người khác thì nhắn họ hoặc mở PR.
- Không commit `.env`, `mlflow.db`, `mlruns/`, `reports/`, khóa hay token.
- Cập nhật checkbox của mình trong `docs/BUILD_PLAN.md` khi xong.
- Dữ liệu và số liệu giả lập không được trình bày như kết quả thật.

## Rủi ro
- **Docker Desktop** phải chạy trên máy bạn và Teammate 1 (cần để thử Prometheus / Grafana).
- **Evidently và Python 3.14**: có thể chưa hỗ trợ; image dùng 3.11.
- **Chờ nhau**: Teammate 1 và 2 viết code không cần Docker, chỉ phần chạy thử stack mới cần Stage E xong khung compose.
