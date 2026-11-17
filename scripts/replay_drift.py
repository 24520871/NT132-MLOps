"""Run a deterministic, simulated feature-drift replay without external data."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops.monitoring.drift import (
    DriftResult,
    detect_drift,
    write_html_report,
    write_prometheus_metrics,
)
from mlops.monitoring.perf_monitor import PerformanceResult, evaluate_performance

DEFAULT_REPORT_DIR = Path("reports/drift")
DEFAULT_METRICS_PATH = Path("reports/metrics/drift.prom")
DEFAULT_PREDICTION_LOG = Path("reports/prediction_log/predictions.jsonl")


def generate_replay_frames(
    *,
    rows: int = 500,
    seed: int = 42,
    shift_strength: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return reference, unshifted-current, and shifted-current synthetic features."""
    if rows < 2:
        raise ValueError("rows must be at least 2")
    if not np.isfinite(shift_strength) or shift_strength <= 0:
        raise ValueError("shift_strength must be a finite positive number")

    rng = np.random.default_rng(seed)

    def sample() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "tenure": np.clip(rng.normal(36, 12, rows), 0, 72),
                "MonthlyCharges": np.clip(rng.normal(65, 18, rows), 18, 120),
                "Contract": rng.choice(
                    ["Month-to-month", "One year", "Two year"],
                    size=rows,
                    p=[0.55, 0.25, 0.20],
                ),
            }
        )

    reference = sample()
    normal_current = sample()
    shifted_current = sample()
    shifted_current["tenure"] = np.clip(
        rng.normal(36 + 10 * shift_strength, 8, rows),
        0,
        72,
    )
    shifted_current["MonthlyCharges"] = np.clip(
        rng.normal(65 + 12 * shift_strength, 10, rows),
        18,
        120,
    )
    month_to_month_share = min(0.55 + 0.15 * shift_strength, 0.98)
    remaining = 1 - month_to_month_share
    shifted_current["Contract"] = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=rows,
        p=[month_to_month_share, remaining * 0.55, remaining * 0.45],
    )
    return reference, normal_current, shifted_current


def _performance_summary(log_path: Path) -> PerformanceResult:
    if not log_path.exists():
        return PerformanceResult(None, None, 0, "prediction log is not present")
    return evaluate_performance(log_path)


def _drift_summary(result: DriftResult) -> dict[str, Any]:
    return {
        "data_drift_share": result.data_drift_share,
        "data_drift_detected": int(result.data_drift_detected),
        "drifted_features": list(result.drifted_features),
        "feature_count": len(result.feature_results),
    }


def run_replay(
    *,
    rows: int = 500,
    seed: int = 42,
    shift_strength: float = 2.0,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    prediction_log_path: str | Path = DEFAULT_PREDICTION_LOG,
    p_value_threshold: float = 0.05,
    drift_share_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run normal and shifted comparisons, writing reports and current Prometheus gauges."""
    reference, normal_current, shifted_current = generate_replay_frames(
        rows=rows,
        seed=seed,
        shift_strength=shift_strength,
    )
    normal_result = detect_drift(
        reference,
        normal_current,
        p_value_threshold=p_value_threshold,
        drift_share_threshold=drift_share_threshold,
    )
    shifted_result = detect_drift(
        reference,
        shifted_current,
        p_value_threshold=p_value_threshold,
        drift_share_threshold=drift_share_threshold,
    )

    report_dir = Path(report_dir)
    normal_report = write_html_report(
        normal_result,
        report_dir / "replay-normal.html",
        simulated=True,
    )
    shifted_report = write_html_report(
        shifted_result,
        report_dir / "replay-shifted.html",
        simulated=True,
    )
    performance = _performance_summary(Path(prediction_log_path))
    metrics = write_prometheus_metrics(
        shifted_result,
        metrics_path,
        simulated=True,
        performance=performance,
    )

    return {
        "simulation": "simulated",
        "seed": seed,
        "rows_per_dataset": rows,
        "shift_strength": shift_strength,
        "normal": _drift_summary(normal_result),
        "shifted": _drift_summary(shifted_result),
        "performance": asdict(performance),
        "reports": {
            "normal": str(normal_report),
            "shifted": str(shifted_report),
        },
        "metrics": str(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate normal and shifted feature data and export drift results."
    )
    parser.add_argument("--rows", type=int, default=500, help="Rows in each synthetic dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--shift-strength",
        type=float,
        default=2.0,
        help="Magnitude of the synthetic distribution shift",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for generated HTML reports (default: reports/drift)",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Prometheus textfile output path (default: reports/metrics/drift.prom)",
    )
    parser.add_argument(
        "--prediction-log",
        type=Path,
        default=DEFAULT_PREDICTION_LOG,
        help="Optional source of already-labeled prediction records",
    )
    parser.add_argument(
        "--p-value-threshold",
        type=float,
        default=0.05,
        help="Per-feature statistical significance threshold",
    )
    parser.add_argument(
        "--drift-share-threshold",
        type=float,
        default=0.5,
        help="Share of drifted features needed to flag dataset-level drift",
    )
    args = parser.parse_args()

    summary = run_replay(
        rows=args.rows,
        seed=args.seed,
        shift_strength=args.shift_strength,
        report_dir=args.report_dir,
        metrics_path=args.metrics_path,
        prediction_log_path=args.prediction_log,
        p_value_threshold=args.p_value_threshold,
        drift_share_threshold=args.drift_share_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
