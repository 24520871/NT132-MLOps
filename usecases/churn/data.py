"""Telco churn data: load the real CSV, clean, validate, split. Fails loudly if data is missing."""
import pandas as pd
from sklearn.model_selection import train_test_split

from mlops.data import ingest, validation
from usecases.churn.schema import FEATURE_COLUMNS, ID_COLUMNS, SCHEMA, TARGET


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw CSV -> model-ready frame (features + 0/1 target)."""
    validation.check_columns(raw, ID_COLUMNS + FEATURE_COLUMNS + [TARGET])
    df = raw.drop(columns=ID_COLUMNS).copy()
    # TotalCharges is text in the source; blanks are customers with tenure 0 -> no charges yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    if df[TARGET].isna().any():
        raise validation.DataValidationError([f"'{TARGET}' must be Yes/No"])
    df[TARGET] = df[TARGET].astype(int)
    return df[FEATURE_COLUMNS + [TARGET]]


def load_data(cfg: dict) -> pd.DataFrame:
    """Read, clean and validate the Telco dataset described by cfg['data']."""
    df = clean(ingest.read_csv(ingest.locate(cfg["data"])))
    validation.validate(df, SCHEMA)
    return df


def split_data(df: pd.DataFrame, cfg: dict):
    """Stratified train/test split with a fixed seed; nothing is fitted before this point."""
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    return train_test_split(
        X, y, test_size=cfg["split"]["test_size"], stratify=y, random_state=cfg["split"]["seed"]
    )
