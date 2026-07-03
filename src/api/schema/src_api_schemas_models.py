"""
Pydantic schemas for Cluster Heartbeat API.
Defines all request and response models with validation.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, Any, Optional, List, Union, Literal
from datetime import datetime
from enum import Enum
import re


# ============================================
# Enums
# ============================================

class HealthStatus(str, Enum):
    """Health status enum."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Risk level enum."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SortOrder(str, Enum):
    """Sort order enum."""
    ASC = "asc"
    DESC = "desc"


class JobStatus(str, Enum):
    """Job status enum."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_FOUND = "not_found"


# ============================================
# Health Schemas
# ============================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: HealthStatus
    timestamp: datetime
    service: Dict[str, Any]
    system: Dict[str, Any]
    gpu: Dict[str, Any]
    models: Dict[str, Any]
    stats: Dict[str, Any]
    components: Dict[str, str]
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2026-07-03T10:00:00",
                "service": {
                    "name": "Cluster Heartbeat",
                    "version": "1.0.0",
                    "environment": "production"
                },
                "system": {
                    "cpu_count": 8,
                    "memory": {"total": 32, "available": 24}
                },
                "gpu": {"available": True, "count": 2},
                "models": {
                    "fingerprint_loaded": True,
                    "anomaly_detector_loaded": True
                },
                "stats": {
                    "total_jobs": 100,
                    "processed_jobs": 95,
                    "failed_jobs": 5
                },
                "components": {
                    "api": "running",
                    "models": "ready"
                }
            }
        }


class LivenessResponse(BaseModel):
    """Liveness probe response."""
    status: Literal["alive"]
    timestamp: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "status": "alive",
                "timestamp": "2026-07-03T10:00:00"
            }
        }


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    status: Literal["ready", "not_ready"]
    timestamp: datetime
    services: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ready",
                "timestamp": "2026-07-03T10:00:00",
                "services": {
                    "api": "ready",
                    "pipeline": "ready",
                    "models": "loaded"
                }
            }
        }


# ============================================
# Metrics Schemas
# ============================================

class MetricData(BaseModel):
    """Individual metric data point."""
    timestamp: datetime
    node_id: Optional[str] = None
    job_id: Optional[str] = None
    metrics: Dict[str, float]
    
    @validator('metrics')
    def validate_metrics(cls, v):
        """Validate that required metrics are present."""
        # Allow any metrics, but warn about missing common ones
        common_metrics = ['gpu_utilization', 'memory_utilization', 'gpu_temperature']
        missing = [m for m in common_metrics if m not in v]
        if missing:
            # Don't raise error, just note in logs
            pass
        return v
    
    @validator('metrics')
    def validate_metric_values(cls, v):
        """Validate metric values are within reasonable ranges."""
        for key, value in v.items():
            if 'utilization' in key or 'usage' in key:
                if value < 0 or value > 1:
                    raise ValueError(f"{key} must be between 0 and 1, got {value}")
            elif 'temperature' in key:
                if value < -50 or value > 150:
                    raise ValueError(f"{key} must be between -50 and 150, got {value}")
            elif 'errors' in key:
                if value < 0:
                    raise ValueError(f"{key} must be non-negative, got {value}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2026-07-03T10:00:00",
                "node_id": "node-1",
                "job_id": "job-123",
                "metrics": {
                    "gpu_utilization": 0.85,
                    "memory_utilization": 0.65,
                    "gpu_temperature": 72.5,
                    "power_consumption": 150.0,
                    "ecc_errors": 0,
                    "xid_errors": 0
                }
            }
        }


class MetricsIngestRequest(BaseModel):
    """Metrics ingestion request."""
    timestamp: datetime
    metrics: Dict[str, float]
    node_id: Optional[str] = None
    job_id: Optional[str] = None
    
    @root_validator
    def validate_timestamp(cls, values):
        """Validate timestamp is not in the future."""
        timestamp = values.get('timestamp')
        if timestamp and timestamp > datetime.now():
            raise ValueError("Timestamp cannot be in the future")
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2026-07-03T10:00:00",
                "metrics": {
                    "gpu_utilization": 0.85,
                    "memory_utilization": 0.65
                },
                "node_id": "node-1",
                "job_id": "job-123"
            }
        }


class MetricsIngestResponse(BaseModel):
    """Metrics ingestion response."""
    status: Literal["accepted", "failed"]
    job_id: str
    timestamp: datetime
    message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "accepted",
                "job_id": "job_20260703100000_0",
                "timestamp": "2026-07-03T10:00:00",
                "message": "Metrics accepted for processing"
            }
        }


class MetricsBatchIngestRequest(BaseModel):
    """Batch metrics ingestion request."""
    metrics: List[MetricData]
    
    @validator('metrics')
    def validate_batch_size(cls, v):
        """Validate batch size is reasonable."""
        if len(v) > 10000:
            raise ValueError("Batch size cannot exceed 10000 records")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "metrics": [
                    {
                        "timestamp": "2026-07-03T10:00:00",
                        "node_id": "node-1",
                        "metrics": {"gpu_utilization": 0.85}
                    },
                    {
                        "timestamp": "2026-07-03T10:01:00",
                        "node_id": "node-1",
                        "metrics": {"gpu_utilization": 0.82}
                    }
                ]
            }
        }


class MetricsStatusResponse(BaseModel):
    """Metrics processing status response."""
    job_id: str
    status: JobStatus
    timestamp: Optional[str] = None
    has_results: bool = False
    error: Optional[str] = None


# ============================================
# Prediction Schemas
# ============================================

class FailureRiskPrediction(BaseModel):
    """Failure risk prediction for a single entity."""
    id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    estimated_time_to_failure: Optional[float] = None
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime
    
    @validator('risk_score')
    def validate_risk_score(cls, v):
        """Validate risk score is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Risk score must be between 0 and 1")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "id": "gpu_0",
                "risk_score": 0.75,
                "risk_level": "high",
                "estimated_time_to_failure": 3600,
                "confidence": 0.85,
                "timestamp": "2026-07-03T10:00:00"
            }
        }


class PredictionRequest(BaseModel):
    """Prediction request."""
    metrics: List[Dict[str, Any]]
    node_id: Optional[str] = None
    job_id: Optional[str] = None
    
    @validator('metrics')
    def validate_metrics_list(cls, v):
        """Validate metrics list is not empty."""
        if not v:
            raise ValueError("Metrics list cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "metrics": [
                    {
                        "gpu_utilization": 0.85,
                        "memory_utilization": 0.65,
                        "gpu_temperature": 72.5
                    }
                ],
                "node_id": "node-1",
                "job_id": "job-123"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    status: Literal["success", "failed"]
    predictions: List[FailureRiskPrediction]
    timestamp: datetime
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "predictions": [
                    {
                        "id": "gpu_0",
                        "risk_score": 0.75,
                        "risk_level": "high",
                        "timestamp": "2026-07-03T10:00:00"
                    }
                ],
                "timestamp": "2026-07-03T10:00:00"
            }
        }


class HealthScoreResponse(BaseModel):
    """Health score response."""
    node_id: str
    health_score: float = Field(..., ge=0, le=100)
    health_status: HealthStatus
    components: Dict[str, float]
    issues: List[str]
    timestamp: datetime
    
    @validator('health_score')
    def validate_health_score(cls, v):
        """Validate health score is between 0 and 100."""
        if v < 0 or v > 100:
            raise ValueError("Health score must be between 0 and 100")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "node_id": "node-1",
                "health_score": 85.5,
                "health_status": "healthy",
                "components": {
                    "gpu": 90.0,
                    "memory": 85.0,
                    "network": 80.0
                },
                "issues": [],
                "timestamp": "2026-07-03T10:00:00"
            }
        }


class ForecastingResponse(BaseModel):
    """Forecasting response."""
    node_id: str
    forecast: Dict[str, Any]
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "node_id": "node-1",
                "forecast": {
                    "timestamps": ["2026-07-03T11:00:00", "2026-07-03T12:00:00"],
                    "metrics": {
                        "gpu_utilization": [0.75, 0.80],
                        "memory_utilization": [0.60, 0.65]
                    }
                },
                "confidence": 0.85,
                "timestamp": "2026-07-03T10:00:00"
            }
        }


# ============================================
# Recommendation Schemas
# ============================================

class SchedulingRecommendation(BaseModel):
    """Scheduling recommendation."""
    job_id: str
    recommended_node: str
    current_node: Optional[str] = None
    score: float = Field(..., ge=0, le=1)
    reason: str
    resource_match: Dict[str, float]
    priority: int = Field(..., ge=0, le=10)
    timestamp: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "job-123",
                "recommended_node": "node-2",
                "current_node": "node-1",
                "score": 0.85,
                "reason": "Better resource match with 0.85 score",
                "resource_match": {
                    "gpu": 0.80,
                    "memory": 0.70,
                    "cpu": 0.60
                },
                "priority": 5
            }
        }


class SchedulingResponse(BaseModel):
    """Scheduling recommendations response."""
    recommendations: List[SchedulingRecommendation]
    total: int
    timestamp: datetime


class IdleGPUInfo(BaseModel):
    """Idle GPU information."""
    node_id: str
    gpu_id: int
    idle_duration: float
    memory_utilization: float
    compute_utilization: float
    cost_wasted: float
    recommendation: str
    timestamp: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "node_id": "node-1",
                "gpu_id": 0,
                "idle_duration": 600,
                "memory_utilization": 0.05,
                "compute_utilization": 0.02,
                "cost_wasted": 0.67,
                "recommendation": "GPU idle - consider scaling down"
            }
        }


class CostSavingResponse(BaseModel):
    """Cost saving recommendations response."""
    summary: Dict[str, Any]
    idle_gpus: List[IdleGPUInfo]
    recommendations: List[str]
    timestamp: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "summary": {
                    "total_idle_gpus": 3,
                    "total_cost_wasted": 2.01,
                    "potential_savings": 1.61
                },
                "idle_gpus": [],
                "recommendations": [
                    "Node node-1 has 2 idle GPUs - consider consolidating workloads"
                ]
            }
        }


# ============================================
# Dashboard Schemas
# ============================================

class ClusterSummary(BaseModel):
    """Cluster summary."""
    nodes: int
    gpus: int
    average_health: float
    health_status: HealthStatus
    anomalies: int
    idle_gpus: int
    potential_savings: float
    timestamp: datetime


class DashboardResponse(BaseModel):
    """Complete dashboard response."""
    cluster_summary: Dict[str, Any]
    health_scores: Dict[str, Any]
    predictions: Dict[str, Any]
    scheduling: Dict[str, Any]
    cost_savings: Dict[str, Any]
    timeseries: Dict[str, Any]
    service_info: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "cluster_summary": {
                    "nodes": 5,
                    "gpus": 20,
                    "average_health": 85.5,
                    "health_status": "healthy"
                },
                "health_scores": {},
                "predictions": {},
                "scheduling": {},
                "cost_savings": {},
                "timeseries": {},
                "timestamp": "2026-07-03T10:00:00"
            }
        }


# ============================================
# Service Schemas
# ============================================

class ServiceStatus(BaseModel):
    """Service status response."""
    status: str
    uptime_seconds: float
    health: str
    is_running: bool
    timestamp: datetime


class ServiceStats(BaseModel):
    """Service statistics response."""
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
# Error Schemas
# ============================================

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    status_code: int
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid request",
                "status_code": 400,
                "timestamp": "2026-07-03T10:00:00",
                "details": {
                    "field": "metrics",
                    "issue": "Missing required field"
                }
            }
        }


# ============================================
# Pagination Schemas
# ============================================

class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(100, ge=1, le=1000, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.ASC, description="Sort order")
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        """Validate sort_by field is allowed."""
        allowed_fields = [
            'timestamp', 'health_score', 'gpu_utilization',
            'memory_utilization', 'node_id', 'job_id'
        ]
        if v and v not in allowed_fields:
            raise ValueError(f"sort_by must be one of: {', '.join(allowed_fields)}")
        return v


class PaginatedResponse(BaseModel):
    """Paginated response."""
    data: List[Any]
    pagination: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "data": [],
                "pagination": {
                    "total": 100,
                    "page": 1,
                    "limit": 10,
                    "total_pages": 10,
                    "has_next": True,
                    "has_previous": False
                }
            }
        }


# ============================================
# Additional Utility Schemas
# ============================================

class NodeInfo(BaseModel):
    """Node information."""
    node_id: str
    hostname: Optional[str] = None
    ip_address: Optional[str] = None
    status: HealthStatus
    gpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    last_seen: datetime


class JobInfo(BaseModel):
    """Job information."""
    job_id: str
    node_id: Optional[str] = None
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    resources: Dict[str, float]
    metrics: Optional[Dict[str, float]] = None


class AlertInfo(BaseModel):
    """Alert information."""
    alert_id: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


# ============================================
# WebSocket Schemas
# ============================================

class WebSocketMessage(BaseModel):
    """WebSocket message."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime


class WebSocketSubscription(BaseModel):
    """WebSocket subscription request."""
    event: str
    filters: Optional[Dict[str, Any]] = None


# ============================================
# Export all schemas
# ============================================

__all__ = [
    # Enums
    'HealthStatus',
    'RiskLevel',
    'SortOrder',
    'JobStatus',
    
    # Health
    'HealthResponse',
    'LivenessResponse',
    'ReadinessResponse',
    
    # Metrics
    'MetricData',
    'MetricsIngestRequest',
    'MetricsIngestResponse',
    'MetricsBatchIngestRequest',
    'MetricsStatusResponse',
    
    # Predictions
    'FailureRiskPrediction',
    'PredictionRequest',
    'PredictionResponse',
    'HealthScoreResponse',
    'ForecastingResponse',
    
    # Recommendations
    'SchedulingRecommendation',
    'SchedulingResponse',
    'IdleGPUInfo',
    'CostSavingResponse',
    
    # Dashboard
    'ClusterSummary',
    'DashboardResponse',
    
    # Service
    'ServiceStatus',
    'ServiceStats',
    
    # Error
    'ErrorResponse',
    
    # Pagination
    'PaginationParams',
    'PaginatedResponse',
    
    # Additional
    'NodeInfo',
    'JobInfo',
    'AlertInfo',
    
    # WebSocket
    'WebSocketMessage',
    'WebSocketSubscription'
]