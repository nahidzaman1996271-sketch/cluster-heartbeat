"""
Shapes the inference pipeline's output into a dashboard-ready JSON payload
suitable for Grafana (via its JSON API datasource) or a custom React
dashboard: time-series-friendly panels, alerts, and cluster summary.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from utils.logger import get_logger

log = get_logger("dashboard_json_builder")


def build_dashboard_payload(pipeline_result: Dict) -> Dict:
    """
    pipeline_result: the dict returned by inference.pipeline.ClusterHeartbeatPipeline.run()

    Returns a dashboard-ready payload with:
      - cluster_summary: top-level KPIs
      - gpu_panels: per-GPU rows (table/heatmap friendly)
      - alerts: high/medium risk GPUs surfaced as alert objects
      - recommendations: scheduling + cost, flattened for display
      - generated_at: ISO timestamp
    """
    if pipeline_result.get("status") != "ok":
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": pipeline_result.get("status", "error"),
            "cluster_summary": {},
            "gpu_panels": [],
            "alerts": [],
            "recommendations": {},
        }

    gpus = pipeline_result["gpus"]

    gpu_panels = [
        {
            "gpu_id": g["gpu_id"],
            "timestamp": g["as_of"],
            "health_score": g["gpu_health_score"],
            "failure_risk": g["failure_risk_score"],
            "risk_tier": g["risk_tier"],
            "eta_to_failure_steps": g["estimated_time_to_failure_steps"],
            "anomaly_score": g["anomaly_score"],
            "utilization_pct": g["gpu_util_mean"],
            "max_temp_c": g["gpu_temp_max"],
        }
        for g in gpus
    ]

    alerts: List[Dict] = []
    for g in gpus:
        if g["risk_tier"] == "high":
            alerts.append({
                "severity": "critical",
                "gpu_id": g["gpu_id"],
                "message": f"{g['gpu_id']} has high failure risk "
                           f"({g['failure_risk_score']:.2f}); estimated "
                           f"{g['estimated_time_to_failure_steps']:.0f} steps to failure.",
            })
        elif g["risk_tier"] == "medium":
            alerts.append({
                "severity": "warning",
                "gpu_id": g["gpu_id"],
                "message": f"{g['gpu_id']} shows elevated failure risk "
                           f"({g['failure_risk_score']:.2f}); monitor closely.",
            })

    cost = pipeline_result.get("cost_optimization", {})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "cluster_summary": {
            "cluster_health_score": pipeline_result["cluster_health_score"],
            "num_gpus": pipeline_result["num_gpus"],
            "num_alerts": len(alerts),
            "num_gpus_recommended_for_drain": len(pipeline_result.get("gpus_recommended_for_drain", [])),
            "estimated_idle_savings_usd_per_window": cost.get("estimated_savings_usd_per_window", 0),
            "idle_gpu_fraction": cost.get("idle_fraction", 0),
        },
        "gpu_panels": gpu_panels,
        "alerts": alerts,
        "recommendations": {
            "scheduling": pipeline_result.get("scheduling_recommendations", []),
            "drain_candidates": pipeline_result.get("gpus_recommended_for_drain", []),
            "cost_optimization": cost,
        },
    }
    log.info(f"Built dashboard payload: {len(gpu_panels)} GPU panels, {len(alerts)} alerts")
    return payload


if __name__ == "__main__":
    from pathlib import Path
    from config.settings import SETTINGS
    from data.ingestion import load_trace
    from inference.pipeline import ClusterHeartbeatPipeline
    from utils.helpers import save_json

    raw_path = Path(SETTINGS["paths"]["raw_data_dir"]) / "synthetic_trace.csv"
    raw = load_trace(raw_path)

    pipeline = ClusterHeartbeatPipeline()
    result = pipeline.run(raw)
    payload = build_dashboard_payload(result)

    print("cluster_summary:", payload["cluster_summary"])
    print("num alerts:", len(payload["alerts"]))
    if payload["alerts"]:
        print("sample alert:", payload["alerts"][0])

    out_path = Path(SETTINGS["paths"]["processed_data_dir"]) / "dashboard_payload.json"
    save_json(payload, out_path)
    print(f"Saved dashboard payload to {out_path}")
