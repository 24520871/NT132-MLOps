"""Locating and reading raw data files."""
import hashlib
from pathlib import Path

import pandas as pd

from mlops.config import resolve_path


def locate(data_cfg: dict) -> Path:
    """Return the dataset path or raise a clear error telling the user how to get it."""
    path = resolve_path(data_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Download it from {data_cfg.get('source', 'the dataset source')} and place the CSV "
            "there (or set DATA_PATH). This project never falls back to synthetic data."
        )
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def file_sha256(path: Path) -> str:
    """Content hash recorded in MLflow so every run is tied to an exact dataset version."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
