"""Validate a use case's raw data without training: ingest -> clean -> schema validation.

CLI:  python -m mlops.data.check churn      (non-zero exit + readable error if the data is missing or invalid)
First task of the Airflow DAGs.
"""
import argparse
import importlib

from mlops.config import load_config
from mlops.data.ingest import file_sha256, locate


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the dataset of a use case")
    parser.add_argument("usecase", help="e.g. churn")
    args = parser.parse_args()
    cfg = load_config(args.usecase)
    df = importlib.import_module(f"usecases.{args.usecase}.data").load_data(cfg)  # raises if missing/invalid
    print(f"{args.usecase}: data OK, {len(df)} rows x {df.shape[1]} columns, sha256={file_sha256(locate(cfg['data']))[:12]}")


if __name__ == "__main__":
    main()
