"""FastAPI service. Run:  uvicorn mlops.serving.app:app --port 8000   (docs at /docs)"""
import logging

import pandas as pd
from fastapi import FastAPI, HTTPException

from mlops.config import load_config
from mlops.serving.router import ModelRouter, ModelUnavailable
from mlops.serving.schemas import ChurnRequest, PredictionResponse
from usecases.churn.schema import FEATURE_COLUMNS

log = logging.getLogger(__name__)


def create_app(router: ModelRouter | None = None, cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_config("churn")
    router = router or ModelRouter(cfg["model_name"])
    app = FastAPI(
        title="NT132 MLOps - Churn API",
        description="Predicts customer churn with the registry's current production model.",
    )

    @app.get("/health", summary="Liveness: the process is up")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", summary="Readiness: a production model is loadable")
    def ready() -> dict:
        try:
            _, version = router.get()
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "ready", "model_name": router.model_name, "model_version": version}

    @app.post("/predict/churn", response_model=PredictionResponse, summary="Predict churn")
    def predict_churn(req: ChurnRequest) -> PredictionResponse:
        try:
            model, version = router.get()
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            frame = pd.DataFrame([req.model_dump()])[FEATURE_COLUMNS]
            proba = float(model.predict_proba(frame)[0, 1])
        except Exception as exc:
            log.exception("prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed; see server logs.") from exc
        pred = int(proba >= cfg["threshold"])
        return PredictionResponse(
            prediction=pred,
            label="churn" if pred else "no_churn",
            probability=proba,
            threshold=cfg["threshold"],
            model_name=router.model_name,
            model_version=version,
        )

    return app


app = create_app()
