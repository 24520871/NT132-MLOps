"""Expected schema of the cleaned IBM Telco churn data (features only, target kept separate)."""
from typing import Any

YES_NO = ["Yes", "No"]
YES_NO_NOINTERNET = ["Yes", "No", "No internet service"]

TARGET = "Churn"          # raw values Yes/No, mapped to 1/0 by cleaning
ID_COLUMNS = ["customerID"]

SCHEMA: dict[str, Any] = {
    "min_rows": 100,
    "columns": {
        "gender": {"kind": "categorical", "allowed": ["Female", "Male"]},
        "SeniorCitizen": {"kind": "categorical", "allowed": [0, 1]},
        "Partner": {"kind": "categorical", "allowed": YES_NO},
        "Dependents": {"kind": "categorical", "allowed": YES_NO},
        "tenure": {"kind": "numeric", "min": 0, "max": 72},
        "PhoneService": {"kind": "categorical", "allowed": YES_NO},
        "MultipleLines": {"kind": "categorical", "allowed": ["Yes", "No", "No phone service"]},
        "InternetService": {"kind": "categorical", "allowed": ["DSL", "Fiber optic", "No"]},
        "OnlineSecurity": {"kind": "categorical", "allowed": YES_NO_NOINTERNET},
        "OnlineBackup": {"kind": "categorical", "allowed": YES_NO_NOINTERNET},
        "DeviceProtection": {"kind": "categorical", "allowed": YES_NO_NOINTERNET},
        "TechSupport": {"kind": "categorical", "allowed": YES_NO_NOINTERNET},
        "StreamingTV": {"kind": "categorical", "allowed": YES_NO_NOINTERNET},
        "StreamingMovies": {"kind": "categorical", "allowed": YES_NO_NOINTERNET},
        "Contract": {"kind": "categorical", "allowed": ["Month-to-month", "One year", "Two year"]},
        "PaperlessBilling": {"kind": "categorical", "allowed": YES_NO},
        "PaymentMethod": {
            "kind": "categorical",
            "allowed": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        },
        "MonthlyCharges": {"kind": "numeric", "min": 0, "max": 200},
        "TotalCharges": {"kind": "numeric", "min": 0, "max": 10000},
    },
}

FEATURE_COLUMNS = list(SCHEMA["columns"])
NUMERIC_COLUMNS = [c for c, s in SCHEMA["columns"].items() if s["kind"] == "numeric"]
CATEGORICAL_COLUMNS = [c for c, s in SCHEMA["columns"].items() if s["kind"] == "categorical"]
