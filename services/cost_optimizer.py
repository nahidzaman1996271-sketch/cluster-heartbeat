"""
GPU Cost Optimization service: flags idle/underutilized GPUs and estimates
potential savings from consolidating workloads or powering down idle
capacity, using a configurable hourly cost-per-GPU assumption.
"""
from __future__ import annotations

from typing import List, Dict

import pandas as pd

from utils.logger import get_logger

log = get_logger("cost_optimizer")

DEFAULT_HOURLY_COST_USD = 2.50  # illustrative on-demand cloud GPU price; override as needed
IDLE_UTIL_THRESHOLD = 5.0        # gpu_util_mean below this = considered idle


def detect_idle_gpus(fingerprint_df: pd.DataFrame, idle_threshold: float = IDLE_UTIL_THRESHOLD) -> pd.DataFrame:
    idle = fingerprint_df[fingerprint_df["gpu_util_mean"] < idle_threshold].copy()
    log.info(f"Detected {len(idle)} idle GPUs (util < {idle_threshold}%)")
    return idle


def estimate_cost_savings(fingerprint_df: pd.DataFrame, hourly_cost_usd: float = DEFAULT_HOURLY_COST_USD,
                           idle_threshold: float = IDLE_UTIL_THRESHOLD, window_hours: float = 1.0) -> Dict:
    idle_df = detect_idle_gpus(fingerprint_df, idle_threshold)
    num_idle = len(idle_df)
    potential_savings = num_idle * hourly_cost_usd * window_hours

    recommendations: List[Dict] = []
    for _, row in idle_df.iterrows():
        recommendations.append({
            "gpu_id": row["gpu_id"],
            "gpu_util_mean": round(float(row["gpu_util_mean"]), 2),
            "suggestion": "Consider powering down or reallocating - sustained low utilization detected",
            "estimated_hourly_savings_usd": round(hourly_cost_usd, 2),
        })

    summary = {
        "num_idle_gpus": num_idle,
        "total_gpus": len(fingerprint_df),
        "idle_fraction": round(num_idle / max(len(fingerprint_df), 1), 4),
        "estimated_savings_usd_per_window": round(potential_savings, 2),
        "window_hours": window_hours,
        "recommendations": recommendations,
    }
    log.info(f"Cost optimization: {num_idle}/{len(fingerprint_df)} idle GPUs, "
             f"est. savings ${potential_savings:.2f}/window")
    return summary
