"""
End-to-end inference pipeline: raw telemetry -> unified workload
fingerprint -> health/risk scores -> scheduling & cost recommendations ->
dashboard-ready JSON.

Loads every checkpoint (scaler, PCA embedder or autoencoder, anomaly
detector) once at construction, then serves repeated `.run()` calls
cheaply - this is what the FastAPI layer wraps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import SETTINGS
from utils.logger import get_logger
from data.preprocessing import preprocess
from features.windowing import make_windows, window_summary_stats
from features.normalization import WindowScaler
from models.pca_embedding import PCAEmbedder
from models.anomaly_detector import EmbeddingAnomalyDetector
from services.health_score import compute_health_score, cluster_health_score, risk_tier
from services.scheduler import recommend_scheduling, identify_gpus_to_drain
from services.cost_optimizer import estimate_cost_savings

log = get_logger("inference_pipeline")


class ClusterHeartbeatPipeline:
    """Loads all trained artifacts once; call `.run(raw_df)` per request."""

    def __init__(self, use_autoencoder: bool = False):
        self.use_autoencoder = use_autoencoder
        self.scaler = WindowScaler.load()
        self.embedder = PCAEmbedder.load()
        self.anomaly_detector = EmbeddingAnomalyDetector.load()

        self._autoencoder = None
        if use_autoencoder:
            self._autoencoder = self._load_autoencoder()

        log.info("ClusterHeartbeatPipeline initialized (all checkpoints loaded)")

    def _load_autoencoder(self):
        import torch
        from models.autoencoder import GPUAutoencoder

        ckpt_path = Path(SETTINGS["paths"]["checkpoint_dir"]) / "autoencoder.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = GPUAutoencoder(ckpt["window_size"], ckpt["num_features"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

    def build_fingerprint(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        raw_df: raw telemetry (any subset of GPUs/timerange) matching
        utils.schema.ALL_COLUMNS.

        Returns one fingerprint row per GPU (most recent window), containing
        embedding-derived anomaly score + health/risk scores + raw stats.
        """
        clean = preprocess(raw_df)
        windows, end_ts, gpu_ids = make_windows(clean)

        if len(windows) == 0:
            log.warning("No windows could be built (insufficient history per GPU)")
            return pd.DataFrame()

        scaled = self.scaler.transform(windows)

        if self.use_autoencoder and self._autoencoder is not None:
            import torch
            with torch.no_grad():
                x = torch.tensor(scaled, dtype=torch.float32)
                errors = self._autoencoder.reconstruction_error(x).numpy()
            from models.anomaly_detector import scores_from_reconstruction_error
            anomaly_scores = scores_from_reconstruction_error(errors)
        else:
            z = self.embedder.transform(scaled)
            anomaly_scores = self.anomaly_detector.score(z)

        stats = window_summary_stats(windows)
        health = compute_health_score(stats, anomaly_scores)
        health["gpu_id"] = gpu_ids
        health["window_end_timestamp"] = end_ts
        health = pd.concat([health, stats], axis=1)

        # keep only the most recent window per GPU as the "current" fingerprint
        latest = (
            health.sort_values(["gpu_id", "window_end_timestamp"])
            .groupby("gpu_id", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        return latest

    def run(self, raw_df: pd.DataFrame, hourly_cost_usd: float = 2.50) -> dict:
        """Full pipeline: raw -> fingerprint -> scores -> recommendations -> dict."""
        fingerprint = self.build_fingerprint(raw_df)
        if fingerprint.empty:
            return {"status": "insufficient_data", "gpus": []}

        fingerprint["risk_tier"] = fingerprint["failure_risk_score"].apply(risk_tier)

        gpu_results = []
        for _, row in fingerprint.iterrows():
            gpu_results.append({
                "gpu_id": row["gpu_id"],
                "as_of": str(row["window_end_timestamp"]),
                "gpu_health_score": round(float(row["gpu_health_score"]), 2),
                "failure_risk_score": round(float(row["failure_risk_score"]), 4),
                "risk_tier": row["risk_tier"],
                "estimated_time_to_failure_steps": round(float(row["estimated_time_to_failure_steps"]), 1),
                "anomaly_score": round(float(row["anomaly_score"]), 4),
                "gpu_util_mean": round(float(row["gpu_util_mean"]), 2),
                "gpu_temp_max": round(float(row["gpu_temp_max"]), 2),
            })

        result = {
            "status": "ok",
            "cluster_health_score": round(cluster_health_score(fingerprint), 2),
            "num_gpus": len(fingerprint),
            "gpus": gpu_results,
            "scheduling_recommendations": recommend_scheduling(fingerprint),
            "gpus_recommended_for_drain": identify_gpus_to_drain(fingerprint),
            "cost_optimization": estimate_cost_savings(fingerprint, hourly_cost_usd=hourly_cost_usd),
        }
        return result


if __name__ == "__main__":
    from data.ingestion import load_trace
    from utils.helpers import save_json

    raw_path = Path(SETTINGS["paths"]["raw_data_dir"]) / "synthetic_trace.csv"
    raw = load_trace(raw_path)

    pipeline = ClusterHeartbeatPipeline(use_autoencoder=False)
    result = pipeline.run(raw)
    print("status:", result["status"])
    print("cluster_health_score:", result["cluster_health_score"])
    print("num_gpus:", result["num_gpus"])
    print("sample gpu result:", result["gpus"][0])

    out_path = Path(SETTINGS["paths"]["processed_data_dir"]) / "sample_inference_result.json"
    save_json(result, out_path)
    print(f"Saved sample result to {out_path}")
