"""Model router: serves whichever registry version currently has the `production` alias."""
import logging
import os
import threading
import time
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlops.config import tracking_uri

log = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 30.0


class ModelUnavailable(RuntimeError):
    """No usable production model; the message tells the operator what to do."""


class ModelRouter:
    def __init__(self, model_name: str, alias: str = "production", poll_seconds: float | None = None):
        """`poll_seconds`: how long a looked-up alias is trusted before asking MLflow again
        (env MODEL_POLL_SECONDS, default 30; 0 = check on every call). `reload()` forces a check."""
        self.model_name = model_name
        self.alias = alias
        self.poll_seconds = (
            float(os.getenv("MODEL_POLL_SECONDS", DEFAULT_POLL_SECONDS)) if poll_seconds is None else poll_seconds
        )
        self._model: Any = None
        self._version: str | None = None
        self._checked_at = 0.0
        self._lock = threading.Lock()

    def reload(self) -> tuple[Any, str]:
        """Re-check the alias right now (used by POST /admin/reload after a promote or rollback)."""
        return self.get(force=True)

    def get(self, force: bool = False) -> tuple[Any, str]:
        """Return (model, version). Reloads when the alias has moved to a new version."""
        with self._lock:
            if not force and self._model is not None and time.monotonic() - self._checked_at < self.poll_seconds:
                return self._model, self._version
        uri = tracking_uri()
        mlflow.set_tracking_uri(uri)
        try:
            mv = MlflowClient(uri).get_model_version_by_alias(self.model_name, self.alias)
        except MlflowException as exc:
            raise ModelUnavailable(
                f"No '{self.alias}' model for '{self.model_name}'. Train and promote a model "
                f"by running Gate 1, Gate 2 and guarded promotion. "
                f"({exc.message})"
            ) from exc
        version = str(mv.version)
        with self._lock:
            if version != self._version:
                try:
                    self._model = mlflow.sklearn.load_model(f"models:/{self.model_name}/{version}")
                except Exception as exc:
                    raise ModelUnavailable(
                        f"Model '{self.model_name}' v{version} could not be loaded: {exc}"
                    ) from exc
                self._version = version
                log.info("Loaded %s v%s (%s)", self.model_name, version, self.alias)
            self._checked_at = time.monotonic()
            return self._model, self._version
