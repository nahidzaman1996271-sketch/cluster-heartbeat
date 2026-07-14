"""
Feature normalization: fits a per-feature StandardScaler on training windows
and persists it to checkpoints/ so inference uses the exact same scaling.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from config.settings import SETTINGS
from utils.logger import get_logger
from utils.helpers import ensure_dir

log = get_logger("normalization")

DEFAULT_SCALER_PATH = Path(SETTINGS["paths"]["checkpoint_dir"]) / "feature_scaler.pkl"


class WindowScaler:
    """
    Wraps a sklearn StandardScaler to operate on (N, T, F) window tensors by
    flattening to (N*T, F), fitting/transforming, then reshaping back.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._num_features: int | None = None

    def fit(self, windows: np.ndarray) -> "WindowScaler":
        n, t, f = windows.shape
        self._num_features = f
        flat = windows.reshape(-1, f)
        self.scaler.fit(flat)
        log.info(f"Fit StandardScaler on {n} windows ({n * t} timesteps, {f} features)")
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        n, t, f = windows.shape
        flat = windows.reshape(-1, f)
        flat_scaled = self.scaler.transform(flat)
        return flat_scaled.reshape(n, t, f).astype(np.float32)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)

    def inverse_transform(self, windows: np.ndarray) -> np.ndarray:
        n, t, f = windows.shape
        flat = windows.reshape(-1, f)
        flat_inv = self.scaler.inverse_transform(flat)
        return flat_inv.reshape(n, t, f).astype(np.float32)

    def save(self, path: str | Path = DEFAULT_SCALER_PATH) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Saved scaler to {path}")

    @staticmethod
    def load(path: str | Path = DEFAULT_SCALER_PATH) -> "WindowScaler":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info(f"Loaded scaler from {path}")
        return obj


if __name__ == "__main__":
    from pathlib import Path as _P
    from data.ingestion import load_trace
    from features.windowing import make_windows

    path = _P(SETTINGS["paths"]["processed_data_dir"]) / "clean_trace.csv"
    df = load_trace(path)
    windows, _, _ = make_windows(df)

    scaler = WindowScaler()
    scaled = scaler.fit_transform(windows)
    print("scaled mean~0:", np.abs(scaled.mean()) < 0.1)
    print("scaled std~1:", abs(scaled.std() - 1) < 0.1)

    scaler.save()
    reloaded = WindowScaler.load()
    recon = reloaded.transform(windows)
    print("reload matches:", np.allclose(scaled, recon, atol=1e-5))
