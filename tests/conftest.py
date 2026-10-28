"""Shared fixtures: a small schema-valid fake Telco frame and an isolated MLflow store.

The fake frame exists only for fast, deterministic tests; real training uses the real CSV.
"""
import copy

import mlflow
import numpy as np
import pandas as pd
import pytest

from mlops.config import load_config
from usecases.churn.schema import SCHEMA, TARGET


def make_raw_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Raw-format Telco-like frame (customerID, text TotalCharges, Yes/No target)."""
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {"customerID": [f"C{i:05d}" for i in range(n)]}
    for col, spec in SCHEMA["columns"].items():
        if spec["kind"] == "categorical":
            data[col] = rng.choice(spec["allowed"], n)
        else:
            data[col] = rng.uniform(spec["min"], min(spec["max"], 100), n).round(2)
    df = pd.DataFrame(data)
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 73, n)
    # learnable signal: short tenure + month-to-month churn more
    score = -0.05 * df["tenure"] + 1.5 * (df["Contract"] == "Month-to-month") + rng.normal(0, 0.5, n)
    df[TARGET] = np.where(score > np.median(score), "Yes", "No")
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[0, "TotalCharges"] = " "  # blank, as in the real file
    return df


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


def small_cfg() -> dict:
    """Real config with tiny models so tests stay fast."""
    cfg = copy.deepcopy(load_config("churn"))
    cfg["models"]["xgboost"].update(n_estimators=20, max_depth=2)
    cfg["models"]["logreg"].update(max_iter=200)
    return cfg


@pytest.fixture
def churn_cfg() -> dict:
    return small_cfg()


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    """Point MLflow at a throwaway SQLite DB + artifact dir for the duration of a test."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", (tmp_path / "artifacts").as_posix())
    mlflow.set_tracking_uri(uri)
    yield uri
