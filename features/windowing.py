"""
Sliding-window feature generation.

Converts a long per-GPU time series into fixed-length overlapping windows
of shape (window_size, num_features), which is what the autoencoder,
classifier and demand predictor all consume. Also computes simple
per-window summary stats used by the workload classifier and health score.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from config.settings import SETTINGS
from utils.logger import get_logger
from utils.schema import RAW_FEATURE_COLUMNS

log = get_logger("windowing")


def make_windows_for_gpu(df_gpu: pd.DataFrame, window_size: int, stride: int
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    df_gpu: rows for a single gpu_id, sorted by timestamp.
    Returns:
        windows: (num_windows, window_size, num_features)
        end_timestamps: (num_windows,) timestamp each window ends at
        gpu_ids: (num_windows,) repeated gpu_id
    """
    values = df_gpu[RAW_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    timestamps = df_gpu["timestamp"].to_numpy()
    gpu_id = df_gpu["gpu_id"].iloc[0]

    n = len(values)
    if n < window_size:
        return (np.empty((0, window_size, len(RAW_FEATURE_COLUMNS)), dtype=np.float32),
                np.empty((0,), dtype=timestamps.dtype),
                np.empty((0,), dtype=object))

    starts = range(0, n - window_size + 1, stride)
    windows = np.stack([values[s:s + window_size] for s in starts])
    end_ts = np.array([timestamps[s + window_size - 1] for s in starts])
    gpu_ids = np.array([gpu_id] * len(starts), dtype=object)
    return windows, end_ts, gpu_ids


def make_windows(df: pd.DataFrame, window_size: int | None = None, stride: int | None = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows across every GPU in the DataFrame."""
    cfg = SETTINGS["features"]["sliding_window"]
    window_size = window_size or cfg["window_size"]
    stride = stride or cfg["stride"]

    all_windows, all_ts, all_ids = [], [], []
    for gpu_id, group in df.groupby("gpu_id", sort=False):
        group = group.sort_values("timestamp")
        w, ts, ids = make_windows_for_gpu(group, window_size, stride)
        if len(w) == 0:
            log.warning(f"GPU {gpu_id}: only {len(group)} rows, < window_size {window_size}; skipped")
            continue
        all_windows.append(w)
        all_ts.append(ts)
        all_ids.append(ids)

    windows = np.concatenate(all_windows, axis=0) if all_windows else np.empty((0, window_size, len(RAW_FEATURE_COLUMNS)))
    end_ts = np.concatenate(all_ts, axis=0) if all_ts else np.empty((0,))
    gpu_ids = np.concatenate(all_ids, axis=0) if all_ids else np.empty((0,), dtype=object)

    log.info(f"Generated {len(windows)} windows (size={window_size}, stride={stride}) "
             f"from {df['gpu_id'].nunique()} GPUs")
    return windows, end_ts, gpu_ids


def window_summary_stats(windows: np.ndarray) -> pd.DataFrame:
    """
    Per-window summary features (mean/std/max per raw column) used by the
    workload classifier as a lightweight complement to the learned embedding.
    windows: (N, T, F)
    """
    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    maxs = windows.max(axis=1)
    cols = RAW_FEATURE_COLUMNS
    data = {}
    for i, c in enumerate(cols):
        data[f"{c}_mean"] = means[:, i]
        data[f"{c}_std"] = stds[:, i]
        data[f"{c}_max"] = maxs[:, i]
    return pd.DataFrame(data)


if __name__ == "__main__":
    from pathlib import Path
    from data.ingestion import load_trace

    path = Path(SETTINGS["paths"]["processed_data_dir"]) / "clean_trace.csv"
    df = load_trace(path)
    windows, end_ts, gpu_ids = make_windows(df)
    print("windows shape:", windows.shape)
    stats = window_summary_stats(windows)
    print(stats.shape)
    print(stats.head(2))
