"""FastAPI application exposing fingerprints, scores and recommendations.

Run::

    python scripts/serve.py --ckpt checkpoints/best --data data/synthetic

or::

    CH_CKPT_DIR=checkpoints/best CH_DATA_DIR=data/synthetic \
        uvicorn cluster_heartbeat.api.main:app --port 8000
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from .. import __version__
from ..config import load_config
from ..data.ingestion import IngestResult
from ..data.preprocessing import clean_node_frame
from ..inference.pipeline import HeartbeatPipeline
from ..logging_utils import get_logger
from .schemas import HealthResponse, InferenceRequest

logger = get_logger(__name__)

REQUESTS = Counter("cluster_heartbeat_requests_total",
                   "API requests", ["endpoint"])


def create_app(
    ckpt_dir: str = "checkpoints/best",
    data_dir: str | None = "data/synthetic",
    config_path: str = "configs/default.yaml",
) -> FastAPI:
    app = FastAPI(
        title="Cluster Heartbeat API",
        version=__version__,
        description=(
            "Unified GPU-cluster workload fingerprints powering predictive "
            "failure detection, smart scheduling and cost optimization. "
            "All payloads are dashboard-ready JSON."
        ),
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    # -- state --------------------------------------------------------------
    app.state.pipeline = None
    app.state.report = None
    app.state.histories = {}
    app.state.feature_specs = None

    if Path(ckpt_dir).exists():
        try:
            app.state.pipeline = HeartbeatPipeline(ckpt_dir)
        except Exception:  # pragma: no cover - defensive startup
            logger.exception("failed to load checkpoint at %s", ckpt_dir)

    if app.state.pipeline is not None and data_dir and Path(data_dir).exists():
        try:
            result = IngestResult.load(data_dir)
            cfg = load_config(config_path)
            from ..config import load_feature_registry as _lfr
            specs, _ = _lfr("configs/features.yaml")
            frames = {
                n: clean_node_frame(df, specs, cfg.data.synthetic.interval_seconds)
                for n, df in result.frames.items()
            }
            started = time.perf_counter()
            app.state.report = app.state.pipeline.process_cluster(frames)
            app.state.histories = {
                n: app.state.pipeline.process_node(n, df) for n, df in frames.items()
            }
            logger.info("cluster report built in %.1fs", time.perf_counter() - started)
        except Exception:  # pragma: no cover - defensive startup
            logger.exception("failed to build cluster report from %s", data_dir)

    # -- helpers -------------------------------------------------------------
    def pipeline() -> HeartbeatPipeline:
        if app.state.pipeline is None:
            raise HTTPException(503, "model checkpoint not loaded — train first")
        return app.state.pipeline

    def report() -> dict:
        if app.state.report is None:
            raise HTTPException(503, "cluster report not built — provide --data")
        return app.state.report

    # -- meta ----------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health():
        REQUESTS.labels("/health").inc()
        return HealthResponse(
            status="ok",
            model_loaded=app.state.pipeline is not None,
            backend=app.state.pipeline.backend if app.state.pipeline else None,
            report_loaded=app.state.report is not None,
            version=__version__,
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # -- cluster --------------------------------------------------------------
    @app.get("/api/v1/cluster/summary")
    def cluster_summary():
        REQUESTS.labels("summary").inc()
        r = report()
        return {**r["cluster"], "generated_at": r["generated_at"], "model": r["model"]}

    @app.get("/api/v1/cluster/timeseries")
    def cluster_timeseries():
        REQUESTS.labels("timeseries").inc()
        return report()["timeseries"]

    @app.get("/api/v1/dashboard")
    def dashboard():
        """Everything a dashboard needs in one round trip."""
        REQUESTS.labels("dashboard").inc()
        return report()

    # -- nodes -----------------------------------------------------------------
    @app.get("/api/v1/nodes")
    def nodes():
        REQUESTS.labels("nodes").inc()
        return report()["nodes"]

    @app.get("/api/v1/nodes/{node_id}")
    def node_detail(node_id: str):
        REQUESTS.labels("node_detail").inc()
        r = report()
        if node_id not in r["nodes"]:
            raise HTTPException(404, f"unknown node {node_id!r}")
        node_alerts = [a for a in r["alerts"] if a["node_id"] == node_id]
        return {**r["nodes"][node_id], "node_id": node_id, "alerts": node_alerts}

    @app.get("/api/v1/nodes/{node_id}/fingerprint")
    def node_fingerprint(
        node_id: str,
        tail: int = Query(1, ge=1, le=200,
                          description="how many recent fingerprints to return"),
    ):
        REQUESTS.labels("fingerprint").inc()
        history = app.state.histories.get(node_id)
        if not history:
            raise HTTPException(404, f"no fingerprint history for {node_id!r}")
        return [fp.to_dict() for fp in history[-tail:]]

    # -- alerts & recommendations ------------------------------------------------
    @app.get("/api/v1/alerts")
    def alerts(severity: str | None = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")):
        REQUESTS.labels("alerts").inc()
        items = report()["alerts"]
        if severity:
            items = [a for a in items if a["severity"] == severity]
        return items

    @app.get("/api/v1/recommendations/scheduling")
    def scheduling_recommendation(
        gpu_request: float = Query(0.0, ge=0, le=100),
        mem_request: float = Query(0.0, ge=0, le=100),
    ):
        """Rank nodes for a new workload (Scheduler Framework Score input)."""
        REQUESTS.labels("scheduling").inc()
        from ..services import scheduling

        latest = {n: fps[-1].to_dict() for n, fps in app.state.histories.items() if fps}
        if not latest:
            raise HTTPException(503, "no fingerprints available")
        cfg = load_config(config_path)
        return scheduling.recommend(
            latest, gpu_request=gpu_request, mem_request=mem_request,
            avoid_risk=cfg.scoring.failure["risk_alert_threshold"],
        )

    @app.get("/api/v1/recommendations/cost")
    def cost_recommendation():
        REQUESTS.labels("cost").inc()
        return report()["recommendations"]["cost"]

    # -- explainer (Heartbeat Explainer module) --------------------------------
    @app.get("/api/v1/explainer/cluster")
    def explain_cluster():
        """Explain current cluster state, anomalies, and recommended actions."""
        REQUESTS.labels("explainer_cluster").inc()
        from ..services import explainer
        
        latest = {n: fps[-1].to_dict() for n, fps in app.state.histories.items() if fps}
        if not latest:
            raise HTTPException(503, "no fingerprints available")
        return explainer.explain_cluster(latest)
    
    @app.get("/api/v1/explainer/dashboard")
    def explainer_dashboard():
        """Full Heartbeat Explainer dashboard with shape view, key movers, and suggested actions.
        
        Returns everything needed to render the Explainer panel on the dashboard:
        - Shape view: healthy vs current metrics side-by-side
        - Key movers: top 3 metrics that diverged from baseline
        - Suggested actions: actionable recommendations with reasoning (one per drift)
        - Anomalies: detailed list of detected deviations
        - Cluster state: summary of current health
        """
        REQUESTS.labels("explainer_dashboard").inc()
        from ..services import explainer
        
        latest = {n: fps[-1].to_dict() for n, fps in app.state.histories.items() if fps}
        if not latest:
            raise HTTPException(503, "no fingerprints available")
        return explainer.explain_cluster_with_dashboard(latest)

    @app.get("/api/v1/explainer/node/{node_id}")
    def explain_node(node_id: str):
        """Explain a single node's health, anomalies, and reasoning."""
        REQUESTS.labels("explainer_node").inc()
        from ..services import explainer
        
        if node_id not in app.state.histories or not app.state.histories[node_id]:
            raise HTTPException(404, f"node {node_id} not found")
        
        fingerprint = app.state.histories[node_id][-1].to_dict()
        return explainer.explain_node(node_id, fingerprint)

    # -- ad-hoc inference ----------------------------------------------------------
    @app.post("/api/v1/inference")
    def inference(req: InferenceRequest):
        """Score one telemetry window submitted by a caller (e.g. a collector)."""
        REQUESTS.labels("inference").inc()
        pipe = pipeline()
        df = pd.DataFrame(req.records)
        missing = [f for f in pipe.feature_names if f not in df.columns]
        if missing:
            raise HTTPException(422, f"missing features: {missing}")
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="60s")
        if len(df) != pipe.window_size:
            raise HTTPException(
                422, f"expected exactly {pipe.window_size} rows, got {len(df)}")
        fp = pipe.process_window(req.node_id, df)
        return fp.to_dict()

    # -- dashboard frontend ----------------------------------------------------
    static_dir = Path(__file__).resolve().parents[3] / "static"
    if static_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="dashboard")

    return app


# Module-level app for `uvicorn cluster_heartbeat.api.main:app`.
app = create_app(
    ckpt_dir=os.environ.get("CH_CKPT_DIR", "checkpoints/best"),
    data_dir=os.environ.get("CH_DATA_DIR", "data/synthetic"),
    config_path=os.environ.get("CH_CONFIG", "configs/default.yaml"),
)
