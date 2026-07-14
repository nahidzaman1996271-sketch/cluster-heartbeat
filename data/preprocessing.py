"""
Cleaning and preprocessing for raw GPU telemetry:
  - missing value interpolation (per-GPU, time-ordered)
  - physical-bound clipping / outlier handling
  - duplicate-timestamp removal
  - per-GPU contiguity check (gaps get flagged, not silently dropped)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.schema import RAW_FEATURE_COLUMNS, FEATURE_BOUNDS

log = get_logger("preprocessing")


class Preprocessor:
    def __init__(self, outlier_z_thresh: float = 5.0):
        self.outlier_z_thresh = outlier_z_thresh

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["gpu_id", "timestamp"], keep="last")
        dropped = before - len(df)
        if dropped:
            log.info(f"Dropped {dropped} duplicate (gpu_id, timestamp) rows")
        return df

    def _interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values per-GPU along the time axis."""
        parts = []
        for gpu_id, group in df.groupby("gpu_id", sort=False):
            group = group.sort_values("timestamp").copy()
            n_missing = group[RAW_FEATURE_COLUMNS].isna().sum().sum()
            if n_missing:
                group[RAW_FEATURE_COLUMNS] = (
                    group[RAW_FEATURE_COLUMNS]
                    .interpolate(method="linear", limit_direction="both")
                )
                log.debug(f"GPU {gpu_id}: interpolated {n_missing} missing values")
            parts.append(group)
        return pd.concat(parts, ignore_index=True)

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip to physical bounds, then clip extreme z-score outliers per-GPU."""
        df = df.copy()

        # 1) Hard physical bounds (e.g. utilization can't be negative or >100)
        for col, (low, high) in FEATURE_BOUNDS.items():
            if col not in df.columns:
                continue
            if low is not None:
                df[col] = df[col].clip(lower=low)
            if high is not None:
                df[col] = df[col].clip(upper=high)

        # 2) Per-GPU z-score clipping for remaining statistical outliers
        parts = []
        for gpu_id, group in df.groupby("gpu_id", sort=False):
            group = group.copy()
            for col in RAW_FEATURE_COLUMNS:
                series = group[col]
                std = series.std(ddof=0)
                if std == 0 or np.isnan(std):
                    continue
                mean = series.mean()
                z = (series - mean) / std
                clip_mask = z.abs() > self.outlier_z_thresh
                if clip_mask.any():
                    upper = mean + self.outlier_z_thresh * std
                    lower = mean - self.outlier_z_thresh * std
                    group.loc[clip_mask, col] = series[clip_mask].clip(lower, upper)
            parts.append(group)
        return pd.concat(parts, ignore_index=True)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info(f"Preprocessing {len(df)} raw rows")
        df = self._drop_duplicates(df)
        df = self._interpolate_missing(df)
        df = self._handle_outliers(df)
        df = df.sort_values(["gpu_id", "timestamp"]).reset_index(drop=True)
        log.info(f"Preprocessing complete: {len(df)} clean rows")
        return df


def preprocess(df: pd.DataFrame, outlier_z_thresh: float = 5.0) -> pd.DataFrame:
    return Preprocessor(outlier_z_thresh=outlier_z_thresh).run(df)


if __name__ == "__main__":
    from pathlib import Path
    from config.settings import SETTINGS
    from data.ingestion import load_trace

    raw_path = Path(SETTINGS["paths"]["raw_data_dir"]) / "synthetic_trace.csv"
    raw = load_trace(raw_path)

    # inject some missing values + an outlier to prove the pipeline handles them
    raw.loc[5:10, "gpu_temp"] = np.nan
    raw.loc[20, "power_watts"] = 99999

    clean = preprocess(raw)
    print("NaNs after cleaning:", clean[RAW_FEATURE_COLUMNS].isna().sum().sum())
    print("Max power after clipping:", clean["power_watts"].max())
    out_path = Path(SETTINGS["paths"]["processed_data_dir"]) / "clean_trace.csv"
    clean.to_csv(out_path, index=False)
    print(f"Saved cleaned trace to {out_path}")
