"""
Explainable predictions: 
  1) SHAP-based feature attribution for the anomaly/failure-risk score,
     using a KernelExplainer or TreeExplainer over the IsolationForest.
  2) A transparent, formula-level breakdown of the failure-risk score
     (which is a weighted sum, so it's exactly decomposable without SHAP -
     useful as a lightweight explanation shown directly in the dashboard).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config.settings import SETTINGS
from utils.logger import get_logger

log = get_logger("explainability")


def explain_failure_risk(row: pd.Series) -> Dict:
    """
    Exact decomposition of the failure_risk_score formula from
    services.health_score.compute_health_score, for one fingerprint row.
    Requires the row to also carry the underlying stat columns.
    """
    weights = SETTINGS["scoring"]["health_score_weights"]

    def norm(v, lo, hi):
        hi = max(hi, lo + 1e-8)
        return float(np.clip((v - lo) / (hi - lo), 0, 1))

    ecc_risk = norm(row.get("ecc_errors_max", 0), 0, 10)
    xid_risk = norm(row.get("xid_errors_max", 0), 0, 5)
    ecc_xid_risk = max(ecc_risk, xid_risk)
    thermal_risk = norm(row.get("gpu_temp_max", 0), 70, 100)
    util_stability_risk = norm(row.get("gpu_util_std", 0), 0, 40)
    anomaly_risk = float(row.get("anomaly_score", 0))

    contributions = {
        "anomaly": round(weights["anomaly"] * anomaly_risk, 4),
        "ecc_xid_errors": round(weights["ecc_xid"] * ecc_xid_risk, 4),
        "thermal": round(weights["thermal"] * thermal_risk, 4),
        "utilization_instability": round(weights["utilization_stability"] * util_stability_risk, 4),
    }
    total = round(sum(contributions.values()), 4)
    top_driver = max(contributions, key=contributions.get)

    return {
        "gpu_id": row.get("gpu_id"),
        "failure_risk_score": total,
        "contributions": contributions,
        "top_driver": top_driver,
        "explanation": (
            f"Failure risk of {total:.2f} is primarily driven by '{top_driver}' "
            f"(contributes {contributions[top_driver]:.2f} of the total score)."
        ),
    }


def shap_explain_anomaly(anomaly_detector, embeddings: np.ndarray, embedding_names=None,
                          num_background: int = 100) -> Dict:
    """
    SHAP explanation of the IsolationForest anomaly score over embedding
    dimensions. Requires the `shap` package (see requirements.txt).
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is not installed in this environment. Install it via "
            "`pip install shap` to use SHAP-based explanations."
        ) from exc

    background = embeddings[np.random.choice(
        len(embeddings), size=min(num_background, len(embeddings)), replace=False
    )]
    explainer = shap.Explainer(anomaly_detector.model.decision_function, background)
    shap_values = explainer(embeddings)

    names = embedding_names or [f"latent_{i}" for i in range(embeddings.shape[1])]
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranked = sorted(zip(names, mean_abs), key=lambda kv: -kv[1])

    return {
        "feature_importance": [{"feature": n, "mean_abs_shap": round(float(v), 5)} for n, v in ranked],
        "note": "Higher mean_abs_shap = larger average contribution to the anomaly score across samples.",
    }


if __name__ == "__main__":
    from pathlib import Path
    from data.ingestion import load_trace
    from inference.pipeline import ClusterHeartbeatPipeline

    raw_path = Path(SETTINGS["paths"]["raw_data_dir"]) / "synthetic_trace.csv"
    raw = load_trace(raw_path)

    pipeline = ClusterHeartbeatPipeline()
    fingerprint = pipeline.build_fingerprint(raw)

    row = fingerprint.iloc[0]
    explanation = explain_failure_risk(row)
    print("Formula-based explanation:")
    print(explanation)
