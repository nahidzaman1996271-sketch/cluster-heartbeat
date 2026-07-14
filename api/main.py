"""
FastAPI application exposing Cluster Heartbeat as a REST API.

Endpoints:
  GET  /health              - liveness/readiness check
  POST /infer                - run the full pipeline on submitted telemetry rows
  GET  /dashboard/synthetic  - convenience endpoint: run pipeline on the
                                bundled synthetic trace and return dashboard JSON
  GET  /explain/{gpu_id}     - formula-level failure-risk explanation for one GPU
                                (from the most recent /infer or /dashboard call)

Run with:  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import SETTINGS
from utils.logger import get_logger
from api.schemas import InferenceRequest, InferenceResponse, HealthCheckResponse
from inference.pipeline import ClusterHeartbeatPipeline
from inference.explainability import explain_failure_risk
from dashboards.json_builder import build_dashboard_payload

log = get_logger("api")

app = FastAPI(
    title="Cluster Heartbeat API",
    description="AI-powered GPU cluster monitoring: failure prediction, "
                "smart scheduling, and cost optimization.",
    version=SETTINGS["project"]["version"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded lazily on first request so the app can start even before checkpoints exist.
_pipeline: ClusterHeartbeatPipeline | None = None
_last_fingerprint_cache: pd.DataFrame | None = None


def get_pipeline() -> ClusterHeartbeatPipeline:
    global _pipeline
    if _pipeline is None:
        log.info("Lazily initializing ClusterHeartbeatPipeline")
        _pipeline = ClusterHeartbeatPipeline()
    return _pipeline


@app.get("/health", response_model=HealthCheckResponse)
def health() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok", version=SETTINGS["project"]["version"])


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    global _last_fingerprint_cache
    try:
        df = pd.DataFrame([r.model_dump() for r in request.rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        pipeline = get_pipeline()
        result = pipeline.run(df, hourly_cost_usd=request.hourly_cost_usd)
        _last_fingerprint_cache = pipeline.build_fingerprint(df)
        return InferenceResponse(**result)
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/dashboard/synthetic")
def dashboard_synthetic():
    """Convenience: run the pipeline against the bundled synthetic trace."""
    global _last_fingerprint_cache
    try:
        from data.ingestion import load_trace
        raw_path = Path(SETTINGS["paths"]["raw_data_dir"]) / "synthetic_trace.csv"
        if not raw_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No synthetic trace found - run `python -m data.synthetic_generator` first.",
            )
        raw = load_trace(raw_path)
        pipeline = get_pipeline()
        result = pipeline.run(raw)
        _last_fingerprint_cache = pipeline.build_fingerprint(raw)
        return build_dashboard_payload(result)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Dashboard generation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/explain/{gpu_id}")
def explain(gpu_id: str):
    if _last_fingerprint_cache is None or _last_fingerprint_cache.empty:
        raise HTTPException(
            status_code=404,
            detail="No fingerprint available yet - call /infer or /dashboard/synthetic first.",
        )
    matches = _last_fingerprint_cache[_last_fingerprint_cache["gpu_id"] == gpu_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No cached fingerprint for gpu_id={gpu_id}")
    return explain_failure_risk(matches.iloc[0])
