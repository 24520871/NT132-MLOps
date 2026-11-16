"""FastAPI service. Run:  uvicorn mlops.serving.app:app --port 8000   (docs at /docs)"""
import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST

from mlops.config import load_config
from mlops.serving.metrics import UNMATCHED, Metrics
from mlops.serving.prediction_log import PredictionLog
from mlops.serving.router import ModelRouter, ModelUnavailable
from mlops.serving.schemas import ChurnRequest, PredictionResponse
from usecases.churn.schema import FEATURE_COLUMNS

log = logging.getLogger(__name__)


def create_app(
    router: ModelRouter | None = None, cfg: dict | None = None, prediction_log: PredictionLog | None = None
) -> FastAPI:
    cfg = cfg or load_config("churn")
    router = router or ModelRouter(cfg["model_name"])
    prediction_log = prediction_log or PredictionLog.from_env()  # None = logging disabled
    metrics = Metrics()
    app = FastAPI(
        title="NT132 MLOps - Churn API",
        description="Predicts customer churn with the registry's current production model.",
    )

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        start = time.perf_counter()
        status = 500  # an unhandled exception surfaces as a 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            endpoint = route.path if route is not None else UNMATCHED
            if endpoint != "/metrics":
                metrics.observe_request(endpoint, status, time.perf_counter() - start)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health", summary="Liveness: the process is up")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", summary="Readiness: a production model is loadable")
    def ready() -> dict:
        try:
            _, version = router.get()
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        metrics.set_model(router.model_name, version)
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
        label = "churn" if pred else "no_churn"
        metrics.set_model(router.model_name, version)
        metrics.observe_prediction(label, proba)
        if prediction_log:
            prediction_log.record(req.model_dump(), proba, pred, router.model_name, version)
        return PredictionResponse(
            prediction=pred,
            label=label,
            probability=proba,
            threshold=cfg["threshold"],
            model_name=router.model_name,
            model_version=version,
        )

    return app


app = create_app()
