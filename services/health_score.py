"""
Computes Cluster/GPU Health Score, Failure Risk Score, and Estimated Time
to Failure from the unified workload fingerprint (anomaly score + raw
window stats). Pure numpy/pandas - no torch dependency, so it's usable
even in constrained environments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import SETTINGS
from utils.logger import get_logger

log = get_logger("health_score")


def _normalize(series: pd.Series, lo: float, hi: float) -> pd.Series:
    hi = max(hi, lo + 1e-8)
    return ((series - lo) / (hi - lo)).clip(0, 1)


def compute_health_score(stats: pd.DataFrame, anomaly_score: np.ndarray) -> pd.DataFrame:
    """
    stats: output of features.windowing.window_summary_stats (per-window means/stds/maxs)
    anomaly_score: (N,) array in [0,1], one per window (same order as stats)

    Returns a DataFrame with gpu_health_score, failure_risk_score,
    estimated_time_to_failure_steps, per window.
    """
    weights = SETTINGS["scoring"]["health_score_weights"]

    ecc_risk = _normalize(stats["ecc_errors_max"], 0, 10)
    xid_risk = _normalize(stats["xid_errors_max"], 0, 5)
    ecc_xid_risk = np.maximum(ecc_risk, xid_risk)

    thermal_risk = _normalize(stats["gpu_temp_max"], 70, 100)

    util_stability_risk = _normalize(stats["gpu_util_std"], 0, 40)

    anomaly_risk = pd.Series(anomaly_score, index=stats.index)

    failure_risk = (
        weights["anomaly"] * anomaly_risk +
        weights["ecc_xid"] * ecc_xid_risk +
        weights["thermal"] * thermal_risk +
        weights["utilization_stability"] * util_stability_risk
    ).clip(0, 1)

    health_score = (1.0 - failure_risk) * 100.0

    # Simple heuristic ETA: higher risk -> fewer steps until projected failure.
    # (linear inverse mapping, floor at 1 step, cap at 500 steps "healthy/no imminent risk")
    eta_steps = np.where(
        failure_risk > 0.05,
        np.clip((1.0 - failure_risk) * 500, 1, 500),
        500,
    )

    out = pd.DataFrame({
        "gpu_health_score": health_score,
        "failure_risk_score": failure_risk,
        "estimated_time_to_failure_steps": eta_steps,
        "anomaly_score": anomaly_score,
    }, index=stats.index)
    return out


def risk_tier(failure_risk_score: float) -> str:
    thresholds = SETTINGS["scoring"]["failure_risk_thresholds"]
    if failure_risk_score < thresholds["low"]:
        return "low"
    elif failure_risk_score < thresholds["medium"]:
        return "medium"
    return "high"


def cluster_health_score(per_gpu_scores: pd.DataFrame) -> float:
    """Aggregate cluster-wide health score as the mean of latest per-GPU scores."""
    return float(per_gpu_scores["gpu_health_score"].mean())
