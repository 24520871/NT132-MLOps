"""Append-only prediction log (JSON Lines) that the drift / performance monitors read (Stage H).

One line per served prediction:
  {"ts": "...Z", "model_name": "churn", "model_version": "1", "probability": 0.73,
   "prediction": 1, "features": {"tenure": 5, ...}}

Enabled by env PREDICTION_LOG_PATH (unset = disabled, so tests and plain `uvicorn` write nothing).
Logging must never break serving: any I/O error is logged and swallowed.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class PredictionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "PredictionLog | None":
        path = os.getenv("PREDICTION_LOG_PATH")
        return cls(path) if path else None

    def record(self, features: dict, probability: float, prediction: int, model_name: str, model_version: str) -> None:
        line = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "model_name": model_name,
                "model_version": model_version,
                "probability": probability,
                "prediction": prediction,
                "features": features,
            }
        )
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError:
            log.warning("could not write prediction log at %s", self.path, exc_info=True)


def read_prediction_log(path: str | Path) -> pd.DataFrame:
    """Load the log as a flat frame: ts, model_name, model_version, probability, prediction + one column per feature."""
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, sep=".").rename(columns=lambda c: c.removeprefix("features."))
    df["ts"] = pd.to_datetime(df["ts"])
    return df
