"""
Anomaly detection on top of either the autoencoder's reconstruction error
or an IsolationForest fit on the PCA/autoencoder embedding. Produces a
per-window anomaly score in [0, 1] and a boolean is_anomaly flag.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from config.settings import SETTINGS
from utils.logger import get_logger
from utils.helpers import ensure_dir

log = get_logger("anomaly_detector")

DEFAULT_PATH = Path(SETTINGS["paths"]["checkpoint_dir"]) / "anomaly_detector.pkl"


def scores_from_reconstruction_error(errors: np.ndarray) -> np.ndarray:
    """Min-max normalize reconstruction errors into a [0,1] anomaly score."""
    lo, hi = np.percentile(errors, [1, 99])
    hi = max(hi, lo + 1e-8)
    scaled = (errors - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


class EmbeddingAnomalyDetector:
    """IsolationForest over the learned embedding (PCA or autoencoder latent)."""

    def __init__(self, contamination: float | None = None):
        cfg = SETTINGS["model"]["anomaly_detector"]
        self.contamination = contamination or cfg["contamination"]
        self.model = IsolationForest(
            contamination=self.contamination, random_state=42, n_estimators=200
        )
        self._fitted = False

    def fit(self, embeddings: np.ndarray) -> "EmbeddingAnomalyDetector":
        self.model.fit(embeddings)
        self._fitted = True
        log.info(f"Fit IsolationForest on {len(embeddings)} embeddings "
                 f"(contamination={self.contamination})")
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Higher = more anomalous, scaled to [0,1]."""
        raw = -self.model.score_samples(embeddings)  # sklearn: lower score = more anomalous
        lo, hi = raw.min(), raw.max()
        hi = max(hi, lo + 1e-8)
        return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)

    def predict_is_anomaly(self, embeddings: np.ndarray) -> np.ndarray:
        return self.model.predict(embeddings) == -1

    def save(self, path: str | Path = DEFAULT_PATH) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Saved anomaly detector to {path}")

    @staticmethod
    def load(path: str | Path = DEFAULT_PATH) -> "EmbeddingAnomalyDetector":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    from data.ingestion import load_trace
    from features.windowing import make_windows
    from features.normalization import WindowScaler
    from models.pca_embedding import PCAEmbedder

    path = Path(SETTINGS["paths"]["processed_data_dir"]) / "clean_trace.csv"
    df = load_trace(path)
    windows, _, _ = make_windows(df)
    scaled = WindowScaler().fit_transform(windows)

    emb = PCAEmbedder().fit(scaled)
    z = emb.transform(scaled)

    det = EmbeddingAnomalyDetector().fit(z)
    scores = det.score(z)
    is_anom = det.predict_is_anomaly(z)
    print("score stats: mean=%.3f max=%.3f" % (scores.mean(), scores.max()))
    print("num flagged anomalies:", is_anom.sum(), "/", len(is_anom))
    det.save()
