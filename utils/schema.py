"""
Single source of truth for the 13 raw input features used everywhere in the
pipeline (ingestion, preprocessing, windowing, models). Keeping this in one
place avoids column-order bugs between modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

RAW_FEATURE_COLUMNS: List[str] = [
    "gpu_util",            # % (0-100)
    "mem_util",             # % (0-100)
    "gpu_temp",             # Celsius
    "power_watts",          # Watts
    "ecc_errors",           # count, cumulative-diff per interval
    "xid_errors",           # count, cumulative-diff per interval
    "cpu_util",              # % (0-100)
    "ram_util",              # % (0-100)
    "net_throughput_mbps",  # Mbps
    "disk_io_mbps",          # Mbps
    "job_runtime_s",         # seconds, running job duration
    "queue_length",          # count of queued jobs
    "active_processes",      # count of active processes on the GPU
]

IDENTIFIER_COLUMNS: List[str] = ["timestamp", "gpu_id", "node_id"]

ALL_COLUMNS: List[str] = IDENTIFIER_COLUMNS + RAW_FEATURE_COLUMNS

WORKLOAD_CLASSES: List[str] = ["idle", "light", "steady", "bursty"]

# Reasonable physical bounds used for outlier clipping / validation.
FEATURE_BOUNDS = {
    "gpu_util": (0, 100),
    "mem_util": (0, 100),
    "gpu_temp": (20, 110),
    "power_watts": (0, 700),
    "ecc_errors": (0, None),
    "xid_errors": (0, None),
    "cpu_util": (0, 100),
    "ram_util": (0, 100),
    "net_throughput_mbps": (0, None),
    "disk_io_mbps": (0, None),
    "job_runtime_s": (0, None),
    "queue_length": (0, None),
    "active_processes": (0, None),
}


@dataclass
class GPURecord:
    """One row of raw telemetry for a single GPU at a single timestamp."""
    timestamp: str
    gpu_id: str
    node_id: str
    gpu_util: float
    mem_util: float
    gpu_temp: float
    power_watts: float
    ecc_errors: float
    xid_errors: float
    cpu_util: float
    ram_util: float
    net_throughput_mbps: float
    disk_io_mbps: float
    job_runtime_s: float
    queue_length: float
    active_processes: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()
