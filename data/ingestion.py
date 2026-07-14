"""
Data ingestion layer.

Supports two sources:
  1. File-based ingestion (CSV/Parquet) - synthetic traces or exported
     public datasets (Alibaba Cluster Trace, Google Borg Trace) once
     mapped onto our schema.
  2. Live Prometheus/DCGM ingestion via the Prometheus HTTP query API
     (range queries), for real cluster monitoring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from config.settings import SETTINGS
from utils.logger import get_logger
from utils.schema import ALL_COLUMNS, RAW_FEATURE_COLUMNS

log = get_logger("ingestion")


class FileIngestor:
    """Reads a raw trace file (CSV or Parquet) into a validated DataFrame."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"No such data file: {self.path}")

        if self.path.suffix == ".csv":
            df = pd.read_csv(self.path, parse_dates=["timestamp"])
        elif self.path.suffix in (".parquet", ".pq"):
            df = pd.read_parquet(self.path)
        else:
            raise ValueError(f"Unsupported file extension: {self.path.suffix}")

        missing = set(ALL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Input file missing required columns: {sorted(missing)}")

        log.info(f"Loaded {len(df)} rows from {self.path}")
        return df[ALL_COLUMNS].sort_values(["gpu_id", "timestamp"]).reset_index(drop=True)


class PrometheusIngestor:
    """
    Pulls DCGM-exporter metrics from a live Prometheus instance via the
    range-query API. Maps standard DCGM metric names onto our schema.

    NOTE: requires network access to `base_url`; not exercised in this
    sandbox, but implemented against the real Prometheus HTTP API.
    """

    # Mapping: our column name -> typical DCGM/Prometheus metric name.
    METRIC_MAP = {
        "gpu_util": "DCGM_FI_DEV_GPU_UTIL",
        "mem_util": "DCGM_FI_DEV_MEM_COPY_UTIL",
        "gpu_temp": "DCGM_FI_DEV_GPU_TEMP",
        "power_watts": "DCGM_FI_DEV_POWER_USAGE",
        "ecc_errors": "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
        "xid_errors": "DCGM_FI_DEV_XID_ERRORS",
        "cpu_util": "node_cpu_utilization",
        "ram_util": "node_memory_utilization",
        "net_throughput_mbps": "node_network_receive_bytes_total",
        "disk_io_mbps": "node_disk_io_bytes_total",
        "job_runtime_s": "job_runtime_seconds",
        "queue_length": "scheduler_queue_length",
        "active_processes": "DCGM_FI_DEV_COMPUTE_PIDS",
    }

    def __init__(self, base_url: Optional[str] = None, step: Optional[str] = None):
        cfg = SETTINGS["data"]["prometheus"]
        self.base_url = base_url or cfg["base_url"]
        self.step = step or cfg["query_step"]

    def _range_query(self, promql: str, start: str, end: str) -> pd.DataFrame:
        resp = requests.get(
            f"{self.base_url}/api/v1/query_range",
            params={"query": promql, "start": start, "end": end, "step": self.step},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()["data"]["result"]

        frames = []
        for series in result:
            gpu_id = series["metric"].get("gpu", series["metric"].get("gpu_id", "unknown"))
            node_id = series["metric"].get("instance", series["metric"].get("node", "unknown"))
            values = series["values"]  # list of [timestamp, value]
            df = pd.DataFrame(values, columns=["timestamp", "value"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["value"] = df["value"].astype(float)
            df["gpu_id"] = gpu_id
            df["node_id"] = node_id
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["timestamp", "gpu_id", "node_id", "value"])
        return pd.concat(frames, ignore_index=True)

    def fetch(self, start: str, end: str,
              metrics: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Fetch and pivot all mapped metrics into our wide schema."""
        metrics = list(metrics) if metrics else list(self.METRIC_MAP.keys())
        wide_frames = []

        for col in metrics:
            promql = self.METRIC_MAP[col]
            log.info(f"Querying Prometheus for {col} ({promql})")
            df = self._range_query(promql, start, end)
            if df.empty:
                log.warning(f"No data returned for metric {col}")
                continue
            df = df.rename(columns={"value": col})
            wide_frames.append(df[["timestamp", "gpu_id", "node_id", col]])

        if not wide_frames:
            log.warning("PrometheusIngestor.fetch: no metrics returned any data")
            return pd.DataFrame(columns=ALL_COLUMNS)

        merged = wide_frames[0]
        for df in wide_frames[1:]:
            merged = merged.merge(df, on=["timestamp", "gpu_id", "node_id"], how="outer")

        for col in RAW_FEATURE_COLUMNS:
            if col not in merged.columns:
                merged[col] = float("nan")

        return merged[ALL_COLUMNS].sort_values(["gpu_id", "timestamp"]).reset_index(drop=True)


def load_trace(source: str | Path) -> pd.DataFrame:
    """Convenience entrypoint: load from a file path."""
    return FileIngestor(source).load()


if __name__ == "__main__":
    from config.settings import SETTINGS as _S
    default_path = Path(_S["paths"]["raw_data_dir"]) / "synthetic_trace.csv"
    df = load_trace(default_path)
    print(df.shape)
    print(df.head())
