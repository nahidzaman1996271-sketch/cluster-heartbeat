"""Pydantic request/response models for the FastAPI layer."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class TelemetryRow(BaseModel):
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


class InferenceRequest(BaseModel):
    rows: List[TelemetryRow] = Field(..., description="Raw telemetry rows, ideally >= window_size per GPU")
    hourly_cost_usd: float = Field(2.50, description="Assumed on-demand $/GPU/hour for cost estimates")


class GPUResult(BaseModel):
    gpu_id: str
    as_of: str
    gpu_health_score: float
    failure_risk_score: float
    risk_tier: str
    estimated_time_to_failure_steps: float
    anomaly_score: float
    gpu_util_mean: float
    gpu_temp_max: float


class InferenceResponse(BaseModel):
    status: str
    cluster_health_score: Optional[float] = None
    num_gpus: Optional[int] = None
    gpus: List[GPUResult] = []
    scheduling_recommendations: List[dict] = []
    gpus_recommended_for_drain: List[str] = []
    cost_optimization: dict = {}


class HealthCheckResponse(BaseModel):
    status: str
    version: str
