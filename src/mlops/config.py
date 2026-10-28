"""Settings: per-use-case YAML config plus environment overrides."""
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def load_config(usecase: str) -> dict[str, Any]:
    """Load usecases/<usecase>/config.yaml. DATA_PATH env var overrides data.path."""
    path = PROJECT_ROOT / "usecases" / usecase / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config for use case '{usecase}' at {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if os.getenv("DATA_PATH"):
        cfg["data"]["path"] = os.environ["DATA_PATH"]
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def artifact_root() -> str:
    """Where MLflow stores model files for new experiments (env MLFLOW_ARTIFACT_ROOT, else ./mlruns)."""
    root = os.getenv("MLFLOW_ARTIFACT_ROOT")
    if root and "://" in root:  # e.g. s3://bucket/path
        return root
    return Path(root or PROJECT_ROOT / "mlruns").resolve().as_uri()


def tracking_uri() -> str:
    """MLflow tracking URI (env MLFLOW_TRACKING_URI, default local SQLite in the project root)."""
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    if uri == DEFAULT_TRACKING_URI:  # keep the DB in the project root regardless of cwd
        return f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    return uri
