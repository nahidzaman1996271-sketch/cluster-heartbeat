"""
Smoke tests for the core (non-torch) pipeline: data generation -> ingestion
-> preprocessing -> windowing -> normalization -> PCA embedding -> anomaly
detection -> health scoring -> services -> dashboard JSON.

Run with: pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.synthetic_generator import generate_cluster_trace
from data.preprocessing import preprocess
from features.windowing import make_windows, window_summary_stats
from features.normalization import WindowScaler
from models.pca_embedding import PCAEmbedder
from models.anomaly_detector import EmbeddingAnomalyDetector
from services.health_score import compute_health_score, cluster_health_score, risk_tier
from services.scheduler import recommend_scheduling, identify_gpus_to_drain
from services.cost_optimizer import estimate_cost_savings
from inference.pipeline import ClusterHeartbeatPipeline
from inference.explainability import explain_failure_risk
from dashboards.json_builder import build_dashboard_payload


@pytest.fixture(scope="module")
def small_trace() -> pd.DataFrame:
    return generate_cluster_trace(num_gpus=4, num_days=1, interval_s=60, failure_rate=0.5, seed=7)


def test_synthetic_generation_shape(small_trace):
    assert small_trace["gpu_id"].nunique() == 4
    assert len(small_trace) == 4 * 24 * 60  # 1 day @ 60s


def test_preprocessing_removes_nans(small_trace):
    df = small_trace.copy()
    df.loc[0:5, "gpu_temp"] = np.nan
    clean = preprocess(df)
    assert clean[["gpu_util", "gpu_temp"]].isna().sum().sum() == 0


def test_windowing_shapes(small_trace):
    clean = preprocess(small_trace)
    windows, end_ts, gpu_ids = make_windows(clean, window_size=10, stride=5)
    assert windows.ndim == 3
    assert windows.shape[2] == 13
    assert len(end_ts) == len(gpu_ids) == len(windows)


def test_scaler_roundtrip(small_trace):
    clean = preprocess(small_trace)
    windows, _, _ = make_windows(clean, window_size=10, stride=5)
    scaler = WindowScaler().fit(windows)
    scaled = scaler.transform(windows)
    assert abs(scaled.mean()) < 0.5
    recon = scaler.inverse_transform(scaled)
    assert np.allclose(recon, windows, atol=1e-3)


def test_pca_embedding_and_anomaly(small_trace):
    clean = preprocess(small_trace)
    windows, _, _ = make_windows(clean, window_size=10, stride=5)
    scaler = WindowScaler().fit(windows)
    scaled = scaler.transform(windows)

    embedder = PCAEmbedder(n_components=4).fit(scaled)
    z = embedder.transform(scaled)
    assert z.shape == (len(scaled), 4)

    detector = EmbeddingAnomalyDetector(contamination=0.1).fit(z)
    scores = detector.score(z)
    assert scores.min() >= 0 and scores.max() <= 1


def test_health_score_bounds(small_trace):
    clean = preprocess(small_trace)
    windows, _, _ = make_windows(clean, window_size=10, stride=5)
    stats = window_summary_stats(windows)
    fake_anomaly = np.random.rand(len(stats))
    health = compute_health_score(stats, fake_anomaly)
    assert (health["gpu_health_score"] >= 0).all() and (health["gpu_health_score"] <= 100).all()
    assert (health["failure_risk_score"] >= 0).all() and (health["failure_risk_score"] <= 1).all()
    for v in health["failure_risk_score"]:
        assert risk_tier(v) in ("low", "medium", "high")


def test_services_run_without_error(small_trace):
    clean = preprocess(small_trace)
    windows, _, gpu_ids = make_windows(clean, window_size=10, stride=5)
    stats = window_summary_stats(windows)
    fake_anomaly = np.random.rand(len(stats))
    health = compute_health_score(stats, fake_anomaly)
    health["gpu_id"] = gpu_ids
    health = pd.concat([health, stats], axis=1)
    latest = health.groupby("gpu_id", as_index=False).tail(1)

    recs = recommend_scheduling(latest, top_k=2)
    assert isinstance(recs, list)

    drains = identify_gpus_to_drain(latest, risk_threshold=0.0)  # force at least attempt
    assert isinstance(drains, list)

    savings = estimate_cost_savings(latest)
    assert "estimated_savings_usd_per_window" in savings


def test_full_pipeline_end_to_end(small_trace, tmp_path):
    # Fit + save all artifacts against this small trace so the pipeline
    # can load them (mirrors what training scripts do on the real trace).
    clean = preprocess(small_trace)
    windows, _, _ = make_windows(clean)

    scaler = WindowScaler().fit(windows)
    scaler.save()
    scaled = scaler.transform(windows)

    embedder = PCAEmbedder().fit(scaled)
    embedder.save()
    z = embedder.transform(scaled)

    detector = EmbeddingAnomalyDetector().fit(z)
    detector.save()

    pipeline = ClusterHeartbeatPipeline()
    result = pipeline.run(small_trace)
    assert result["status"] == "ok"
    assert result["num_gpus"] == 4
    assert "cluster_health_score" in result

    payload = build_dashboard_payload(result)
    assert "gpu_panels" in payload
    assert len(payload["gpu_panels"]) == 4

    fingerprint = pipeline.build_fingerprint(small_trace)
    explanation = explain_failure_risk(fingerprint.iloc[0])
    assert "failure_risk_score" in explanation
    assert "top_driver" in explanation
