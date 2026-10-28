"""Model router: serves whichever registry version currently has the `production` alias."""
import logging
import threading
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from mlops.config import tracking_uri

log = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """No usable production model; the message tells the operator what to do."""


class ModelRouter:
    def __init__(self, model_name: str, alias: str = "production"):
        self.model_name = model_name
        self.alias = alias
        self._model: Any = None
        self._version: str | None = None
        self._lock = threading.Lock()

    def get(self) -> tuple[Any, str]:
        """Return (model, version). Reloads automatically when the alias moves to a new version."""
        uri = tracking_uri()
        mlflow.set_tracking_uri(uri)
        try:
            mv = MlflowClient(uri).get_model_version_by_alias(self.model_name, self.alias)
        except MlflowException as exc:
            raise ModelUnavailable(
                f"No '{self.alias}' model for '{self.model_name}'. Train and promote one first: "
                f"`python -m usecases.{self.model_name}.train` then "
                f"`python -m mlops.registry.promote {self.model_name}`. ({exc.message})"
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
            return self._model, self._version
