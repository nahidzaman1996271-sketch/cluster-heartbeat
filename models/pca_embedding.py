"""
PCA-based embedding as a lightweight, fast-to-fit alternative to the
autoencoder for the workload fingerprint. Useful as a baseline / fallback
when torch isn't available, and for explainability (linear components are
easy to interpret vs. a neural net's latent space).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from config.settings import SETTINGS
from utils.logger import get_logger
from utils.helpers import ensure_dir

log = get_logger("pca_embedding")

DEFAULT_PCA_PATH = Path(SETTINGS["paths"]["checkpoint_dir"]) / "pca_embedding.pkl"


class PCAEmbedder:
    def __init__(self, n_components: int | None = None):
        cfg = SETTINGS["model"]["pca"]
        self.n_components = n_components or cfg["n_components"]
        self.pca = PCA(n_components=self.n_components)
        self._fitted = False

    def fit(self, windows: np.ndarray) -> "PCAEmbedder":
        n, t, f = windows.shape
        flat = windows.reshape(n, t * f)
        self.pca.fit(flat)
        self._fitted = True
        explained = self.pca.explained_variance_ratio_.sum()
        log.info(f"Fit PCA with {self.n_components} components, "
                 f"explained variance ratio = {explained:.3f}")
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        n, t, f = windows.shape
        flat = windows.reshape(n, t * f)
        return self.pca.transform(flat)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)

    def reconstruction_error(self, windows: np.ndarray) -> np.ndarray:
        n, t, f = windows.shape
        flat = windows.reshape(n, t * f)
        z = self.pca.transform(flat)
        recon = self.pca.inverse_transform(z)
        return np.mean((recon - flat) ** 2, axis=1)

    def save(self, path: str | Path = DEFAULT_PCA_PATH) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Saved PCA embedder to {path}")

    @staticmethod
    def load(path: str | Path = DEFAULT_PCA_PATH) -> "PCAEmbedder":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    from data.ingestion import load_trace
    from features.windowing import make_windows
    from features.normalization import WindowScaler

    path = Path(SETTINGS["paths"]["processed_data_dir"]) / "clean_trace.csv"
    df = load_trace(path)
    windows, _, _ = make_windows(df)
    scaled = WindowScaler().fit_transform(windows)

    emb = PCAEmbedder()
    z = emb.fit_transform(scaled)
    err = emb.reconstruction_error(scaled)
    print("embedding shape:", z.shape)
    print("recon error stats: mean=%.4f std=%.4f max=%.4f" % (err.mean(), err.std(), err.max()))
    emb.save()
