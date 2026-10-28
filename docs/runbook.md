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

## AWS (Stage F) — TODO
Exact CLI commands and teardown steps will be added here.
