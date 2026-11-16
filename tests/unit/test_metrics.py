"""Metrics contract (docs/TASKS_DASHBOARD.md): names and labels the Grafana dashboard / alerts rely on."""
from prometheus_client.parser import text_string_to_metric_families

from mlops.serving.metrics import Metrics


def samples(m: Metrics) -> dict[tuple[str, tuple], float]:
    out = {}
    for family in text_string_to_metric_families(m.render().decode()):
        for s in family.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def test_request_counter_and_histogram():
    m = Metrics()
    m.observe_request("/predict/churn", 200, 0.02)
    m.observe_request("/predict/churn", 503, 0.01)
    s = samples(m)
    assert s[("api_requests_total", (("endpoint", "/predict/churn"), ("status", "200")))] == 1
    assert s[("api_requests_total", (("endpoint", "/predict/churn"), ("status", "503")))] == 1
    assert s[("api_request_duration_seconds_count", (("endpoint", "/predict/churn"),))] == 2


def test_prediction_metrics():
    m = Metrics()
    m.observe_prediction("churn", 0.83)
    m.observe_prediction("no_churn", 0.12)
    s = samples(m)
    assert s[("churn_predictions_total", (("label", "churn"),))] == 1
    assert s[("churn_predictions_total", (("label", "no_churn"),))] == 1
    assert s[("churn_probability_count", ())] == 2
    assert s[("churn_probability_bucket", (("le", "0.2"),))] == 1  # only 0.12
    assert s[("churn_probability_bucket", (("le", "0.9"),))] == 2  # buckets are cumulative


def test_model_version_info_keeps_only_the_current_version():
    m = Metrics()
    m.set_model("churn", "1")
    m.set_model("churn", "1")  # idempotent
    m.set_model("churn", "2")
    versions = {dict(k[1])["version"] for k in samples(m) if k[0] == "model_version_info"}
    assert versions == {"2"}


def test_instances_do_not_share_a_registry():
    a, b = Metrics(), Metrics()
    a.observe_request("/health", 200, 0.001)
    assert not any(k[0] == "api_requests_total" for k in samples(b))
