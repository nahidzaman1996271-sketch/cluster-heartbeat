"""
Smart GPU Scheduling service: recommends which GPUs are best suited for a
new job (or should be drained), based on health score, failure risk,
current/predicted utilization, and queue length.
"""
from __future__ import annotations

from typing import List, Dict

import pandas as pd

from utils.logger import get_logger

log = get_logger("scheduler")


def recommend_scheduling(fingerprint_df: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """
    fingerprint_df: one row per GPU with at least:
        gpu_id, gpu_health_score, failure_risk_score, gpu_util_mean,
        queue_length_mean, predicted_util_next (optional)

    Returns ranked scheduling recommendations, best candidates first.
    """
    df = fingerprint_df.copy()

    # Score: prefer healthy, low-risk, low-current-utilization, short-queue GPUs.
    df["schedule_score"] = (
        0.4 * (df["gpu_health_score"] / 100.0) +
        0.3 * (1 - df["failure_risk_score"]) +
        0.2 * (1 - df["gpu_util_mean"].clip(0, 100) / 100.0) +
        0.1 * (1 - (df["queue_length_mean"].clip(lower=0) /
                     max(df["queue_length_mean"].max(), 1)))
    )

    df = df.sort_values("schedule_score", ascending=False)

    recommendations = []
    for _, row in df.head(top_k).iterrows():
        recommendations.append({
            "gpu_id": row["gpu_id"],
            "schedule_score": round(float(row["schedule_score"]), 4),
            "reason": _explain_recommendation(row),
        })
    log.info(f"Generated {len(recommendations)} scheduling recommendations")
    return recommendations


def _explain_recommendation(row: pd.Series) -> str:
    parts = []
    if row["gpu_health_score"] >= 80:
        parts.append("high health score")
    if row["failure_risk_score"] < 0.2:
        parts.append("low failure risk")
    if row["gpu_util_mean"] < 30:
        parts.append("currently underutilized")
    if not parts:
        parts.append("best available option among current candidates")
    return ", ".join(parts)


def identify_gpus_to_drain(fingerprint_df: pd.DataFrame, risk_threshold: float = 0.66) -> List[str]:
    """GPUs recommended for draining (no new jobs) due to high failure risk."""
    at_risk = fingerprint_df[fingerprint_df["failure_risk_score"] >= risk_threshold]
    gpu_ids = at_risk["gpu_id"].tolist()
    if gpu_ids:
        log.warning(f"Recommending drain for {len(gpu_ids)} high-risk GPUs: {gpu_ids}")
    return gpu_ids
