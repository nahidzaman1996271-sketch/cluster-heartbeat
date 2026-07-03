"""
Pydantic schemas for API requests and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum


# ============================================
# Health Models
# ============================================

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthResponse(BaseModel):
    status: HealthStatus
    timestamp: datetime
    services: Dict[str, str]
    version: str
    uptime: Optional[float] = None


class LivenessResponse(BaseModel):
    status: str = "alive"


class ReadinessResponse(BaseModel):
    status: str = "ready"
    services: Dict[str, str]


# ============================================
# Metrics Models
# ============================================

class MetricData(BaseModel):
    timestamp: datetime
    node_id: Optional[str] = None
    job_id: Optional[str] = None
    metrics: Dict[str, float]


class MetricsIngestRequest(BaseModel):
    timestamp: datetime
    metrics: Dict[str, float]
    node_id: Optional[str] = None
    job_id: Optional[str] = None
    
    @validator('metrics')
    def validate_metrics(cls, v):
        required = ['gpu_utilization', 'memory_utilization']
        for req in required:
            if req not in v:
                raise ValueError(f"Missing required metric: {req}")
        return v


class MetricsIngestResponse(BaseModel):
    status: str
    job_id: str
    timestamp: datetime


class MetricsBatchIngestRequest(BaseModel):
    metrics: List[MetricData]


# ============================================
# Prediction Models
# ============================================

class FailureRiskPrediction(BaseModel):
    id: str
    risk_score: float
    risk_level: str
    estimated_time_to_failure: Optional[float] = None
    confidence: float
    timestamp: datetime


class PredictionRequest(BaseModel):
    metrics: List[Dict[str, Any]]
    node_id: Optional[str] = None
    job_id: Optional[str] = None


class PredictionResponse(BaseModel):
    status: str
    predictions: List[FailureRiskPrediction]
    timestamp: datetime


class HealthScoreResponse(BaseModel):
    node_id: str
    health_score: float
    health_status: HealthStatus
    components: Dict[str, float]
    issues: List[str]
    timestamp: datetime


class ForecastingResponse(BaseModel):
    node_id: str
    forecast: Dict[str, Any]
    confidence: float
    timestamp: datetime


# ============================================
# Recommendation Models
# ============================================

class SchedulingRecommendation(BaseModel):
    job_id: str
    recommended_node: str
    current_node: Optional[str] = None
    score: float
    reason: str
    resource_match: Dict[str, float]
    priority: int
    timestamp: Optional[float] = None


class SchedulingResponse(BaseModel):
    recommendations: List[SchedulingRecommendation]
    total: int
    timestamp: datetime


class IdleGPUInfo(BaseModel):
    node_id: str
    gpu_id: int
    idle_duration: float
    memory_utilization: float
    compute_utilization: float
    cost_wasted: float
    recommendation: str


class CostSavingResponse(BaseModel):
    total_idle_gpus: int
    total_cost_wasted: float
    idle_gpus: List[IdleGPUInfo]
    potential_savings: float
    recommendations: List[str]
    timestamp: datetime


# ============================================
# Dashboard Models
# ============================================

class ClusterSummary(BaseModel):
    nodes: int
    gpus: int
    average_health: float
    health_status: HealthStatus
    anomalies: int
    idle_gpus: int
    potential_savings: float
    timestamp: datetime


class DashboardResponse(BaseModel):
    cluster_summary: ClusterSummary
    health_scores: Dict[str, Any]
    predictions: Dict[str, Any]
    scheduling: Dict[str, Any]
    cost_savings: Dict[str, Any]
    timeseries: Dict[str, Any]
    timestamp: datetime


# ============================================
# Service Models
# ============================================

class ServiceStatus(BaseModel):
    status: str
    uptime_seconds: float
    health: str
    is_running: bool
    timestamp: datetime


class ServiceStats(BaseModel):
    uptime_seconds: float
    total_jobs: int
    processed_jobs: int
    failed_jobs: int
    avg_processing_time: float
    cache_size: int
    queue_size: int
    is_running: bool
    health_status: str
    last_processed: Optional[datetime] = None


# ============================================
# Error Models
# ============================================

class ErrorResponse(BaseModel):
    error: str
    status_code: int
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


# ============================================
# Pagination Models
# ============================================

class PaginationParams(BaseModel):
    page: int = Field(1, ge=1)
    limit: int = Field(100, ge=1, le=1000)
    sort_by: Optional[str] = None
    sort_order: str = Field("asc", regex="^(asc|desc)$")


class PaginatedResponse(BaseModel):
    data: List[Any]
    pagination: Dict[str, Any]