"""Prometheus metrics for the API (Stage H). Names follow the contract in docs/TASKS_DASHBOARD.md.

Each Metrics instance owns its own registry, so create_app() can be called many times (tests)
without "duplicated timeseries" errors. Drift / recall / AUC gauges are NOT here: they come from
the drift script via the node-exporter textfile collector.
"""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

PROB_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
UNMATCHED = "unmatched"  # endpoint label for 404s: keeps label cardinality bounded


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "api_requests_total", "HTTP requests handled", ["endpoint", "status"], registry=self.registry
        )
        self.duration = Histogram(
            "api_request_duration_seconds", "HTTP request latency", ["endpoint"], registry=self.registry
        )
        self.predictions = Counter(
            "churn_predictions_total", "Churn predictions served", ["label"], registry=self.registry
        )
        self.probability = Histogram(
            "churn_probability", "Predicted churn probability", buckets=PROB_BUCKETS, registry=self.registry
        )
        self.model_info = Gauge(
            "model_version_info", "1 for the model version currently served", ["model", "version"],
            registry=self.registry,
        )
        self._served: tuple[str, str] | None = None

    def observe_request(self, endpoint: str, status: int, seconds: float) -> None:
        self.requests.labels(endpoint=endpoint, status=str(status)).inc()
        self.duration.labels(endpoint=endpoint).observe(seconds)

    def observe_prediction(self, label: str, probability: float) -> None:
        self.predictions.labels(label=label).inc()
        self.probability.observe(probability)

    def set_model(self, model: str, version: str) -> None:
        """Mark `version` as the served one and drop the previous version's series."""
        if self._served == (model, version):
            return
        if self._served is not None:
            self.model_info.remove(*self._served)
        self.model_info.labels(model=model, version=version).set(1)
        self._served = (model, version)

    def render(self) -> bytes:
        return generate_latest(self.registry)
