"""Train the churn models and register the best as `candidate`.

Run from anywhere:  python -m usecases.churn.train
Then gate and promote:  python -m mlops.registry.promote churn
"""
import logging

from mlops.config import load_config, resolve_path
from mlops.data.ingest import file_sha256, locate
from mlops.training.runner import run_training
from usecases.churn.data import load_data, split_data
from usecases.churn.model import build_models


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config("churn")
    df = load_data(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    result = run_training(
        cfg,
        build_models(cfg, y_train),
        X_train,
        X_test,
        y_train,
        y_test,
        data_sha256=file_sha256(locate(cfg["data"])),
    )
    m = {k: round(v, 4) for k, v in result.metrics.items()}
    print(f"best={result.best_model} registered '{cfg['model_name']}' v{result.version} as candidate {m}")
    print(f"run_id={result.run_id}")


if __name__ == "__main__":
    main()
