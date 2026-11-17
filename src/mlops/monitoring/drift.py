"""Feature-level drift checks and Prometheus export for the monitoring dashboard.

The pinned project pandas version currently breaks Evidently 0.7.23 at import time, so
this module uses two-sample tests from the already-installed SciPy dependency.
"""
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, write_to_textfile
from scipy.stats import chi2_contingency, ks_2samp

from mlops.monitoring.perf_monitor import PerformanceResult

TARGET_AND_METADATA = {
    "churn",
    "timestamp",
    "ts",
    "model_name",
    "model_version",
    "prediction",
    "probability",
    "y_true",
    "ground_truth",
}


@dataclass(frozen=True)
class FeatureDrift:
    name: str
    drifted: bool
    p_value: float
    method: str
    reference_rows: int
    current_rows: int


@dataclass(frozen=True)
class DriftResult:
    data_drift_share: float
    data_drift_detected: bool
    feature_results: tuple[FeatureDrift, ...]
    p_value_threshold: float
    drift_share_threshold: float

    @property
    def drifted_features(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.feature_results if item.drifted)


def _feature_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    if not isinstance(reference, pd.DataFrame) or not isinstance(current, pd.DataFrame):
        raise TypeError("reference and current must be pandas DataFrames")
    if reference.empty or current.empty:
        raise ValueError("reference and current datasets must not be empty")
    if not reference.columns.is_unique or not current.columns.is_unique:
        raise ValueError("reference and current must have unique column names")
    if not all(isinstance(col, str) for col in reference.columns) or not all(
        isinstance(col, str) for col in current.columns
    ):
        raise ValueError("all column names must be strings")

    reference_columns = [
        col for col in reference.columns
        if str(col).strip().lower() not in TARGET_AND_METADATA
    ]
    current_columns = [
        col for col in current.columns
        if str(col).strip().lower() not in TARGET_AND_METADATA
    ]
    if not reference_columns:
        raise ValueError("no feature columns remain after excluding target and metadata")

    missing = [col for col in reference_columns if col not in current_columns]
    extra = [col for col in current_columns if col not in reference_columns]
    if missing or extra:
        raise ValueError(f"incompatible feature columns; missing={missing}, extra={extra}")
    return reference_columns


def _validate_thresholds(p_value_threshold: float, drift_share_threshold: float) -> None:
    if not 0 < p_value_threshold < 1:
        raise ValueError("p_value_threshold must be between 0 and 1")
    if not 0 < drift_share_threshold <= 1:
        raise ValueError("drift_share_threshold must be greater than 0 and at most 1")


def _compare_feature(
    name: str,
    reference: pd.Series,
    current: pd.Series,
    p_value_threshold: float,
) -> FeatureDrift:
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) < 2 or len(cur) < 2:
        raise ValueError(f"feature '{name}' needs at least two non-missing values per dataset")

    ref_numeric = (
        pd.api.types.is_numeric_dtype(ref.dtype) and not pd.api.types.is_bool_dtype(ref.dtype)
    )
    cur_numeric = (
        pd.api.types.is_numeric_dtype(cur.dtype) and not pd.api.types.is_bool_dtype(cur.dtype)
    )
    if ref_numeric != cur_numeric:
        raise ValueError(f"feature '{name}' has incompatible numeric/categorical types")

    if ref_numeric:
        ref_values = ref.to_numpy(dtype=float)
        cur_values = cur.to_numpy(dtype=float)
        if not np.isfinite(ref_values).all() or not np.isfinite(cur_values).all():
            raise ValueError(f"numeric feature '{name}' contains infinite values")
        statistic = ks_2samp(ref_values, cur_values, method="auto")
        method = "kolmogorov_smirnov"
    else:
        categories = list(pd.unique(pd.concat([ref, cur], ignore_index=True)))
        ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy()
        cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy()
        if len(categories) < 2:
            p_value = 1.0
        else:
            table = np.vstack([ref_counts, cur_counts])
            p_value = float(chi2_contingency(table, correction=False).pvalue)
        method = "chi_square"
        return FeatureDrift(
            name=name,
            drifted=p_value < p_value_threshold,
            p_value=p_value,
            method=method,
            reference_rows=len(ref),
            current_rows=len(cur),
        )

    p_value = float(statistic.pvalue)
    return FeatureDrift(
        name=name,
        drifted=p_value < p_value_threshold,
        p_value=p_value,
        method=method,
        reference_rows=len(ref),
        current_rows=len(cur),
    )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    p_value_threshold: float = 0.05,
    drift_share_threshold: float = 0.5,
) -> DriftResult:
    """Compare matching feature distributions; target and serving metadata are excluded."""
    _validate_thresholds(p_value_threshold, drift_share_threshold)
    columns = _feature_columns(reference, current)
    results = tuple(
        _compare_feature(
            col,
            reference[col],
            current[col],
            p_value_threshold,
        )
        for col in columns
    )
    share = sum(item.drifted for item in results) / len(results)
    return DriftResult(
        data_drift_share=share,
        data_drift_detected=share >= drift_share_threshold,
        feature_results=results,
        p_value_threshold=p_value_threshold,
        drift_share_threshold=drift_share_threshold,
    )


def write_html_report(
    result: DriftResult,
    output_path: str | Path,
    *,
    simulated: bool = False,
) -> Path:
    """Write a self-contained HTML summary without persisting input records."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(item.name))}</td>"
        f"<td>{'DRIFT' if item.drifted else 'no drift'}</td>"
        f"<td>{item.p_value:.6g}</td>"
        f"<td>{escape(item.method)}</td>"
        f"<td>{item.reference_rows}</td>"
        f"<td>{item.current_rows}</td>"
        "</tr>"
        for item in result.feature_results
    )
    label = "SIMULATED replay data" if simulated else "Observed input data"
    status = "DRIFT DETECTED" if result.data_drift_detected else "No dataset-level drift"
    content = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Feature drift report</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}
table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:.45rem;text-align:left}}
.status{{font-weight:bold}}</style></head>
<body>
<h1>Feature drift report</h1>
<p><strong>Data source:</strong> {escape(label)}</p>
<p class="status">{status}</p>
<p>Drifted feature share: {result.data_drift_share:.4f}
(threshold {result.drift_share_threshold:.4f}); per-feature p-value threshold:
{result.p_value_threshold:.4f}.</p>
<table><thead><tr><th>Feature</th><th>Result</th><th>p-value</th><th>Test</th>
<th>Reference rows</th><th>Current rows</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""
    path.write_text(content, encoding="utf-8")
    return path


def write_prometheus_metrics(
    result: DriftResult,
    output_path: str | Path,
    *,
    simulated: bool = False,
    performance: PerformanceResult | None = None,
) -> Path:
    """Atomically export current drift and available labeled performance metrics."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = CollectorRegistry()
    Gauge("data_drift_share", "Share of input features with detected drift", registry=registry).set(
        result.data_drift_share
    )
    Gauge(
        "data_drift_detected",
        "1 when drifted feature share reaches its threshold",
        registry=registry,
    ).set(int(result.data_drift_detected))
    Gauge(
        "data_drift_simulated",
        "1 when drift metrics come from simulated replay data",
        registry=registry,
    ).set(int(simulated))
    if performance is not None:
        if performance.recall is not None:
            Gauge("model_recall", "Recall on labeled prediction data", registry=registry).set(
                performance.recall
            )
        if performance.auc is not None:
            Gauge("model_auc", "ROC AUC on labeled prediction data", registry=registry).set(
                performance.auc
            )
    write_to_textfile(str(path), registry)
    return path
