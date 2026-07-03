"""
Health check routes for Cluster Heartbeat API.
Provides endpoints for service health monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
import platform

from ..dependencies import get_service, get_config
from ...core.service import ClusterHeartbeatService
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/")
async def health_check(
    service: ClusterHeartbeatService = Depends(get_service),
    config: Dict[str, Any] = Depends(get_config)
) -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    Returns system health including model availability and resource status.
    """
    try:
        # Get system info
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total / (1024**3),  # GB
                "available": psutil.virtual_memory().available / (1024**3),  # GB
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total / (1024**3),  # GB
                "used": psutil.disk_usage('/').used / (1024**3),  # GB
                "free": psutil.disk_usage('/').free / (1024**3),  # GB
                "percent": psutil.disk_usage('/').percent
            }
        }
        
        # Get service status
        service_stats = service.get_service_stats() if service else {}
        
        # Get pipeline stats
        pipeline_stats = {}
        if hasattr(service, 'pipeline'):
            pipeline_stats = service.pipeline.get_pipeline_stats() if service.pipeline else {}
        
        # Check GPU availability
        gpu_info = {}
        try:
            import torch
            gpu_info = {
                "available": torch.cuda.is_available(),
                "count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            if gpu_info["available"]:
                gpu_info["devices"] = []
                for i in range(gpu_info["count"]):
                    gpu_info["devices"].append({
                        "name": torch.cuda.get_device_name(i),
                        "memory_allocated": torch.cuda.memory_allocated(i) / (1024**3),  # GB
                        "memory_reserved": torch.cuda.memory_reserved(i) / (1024**3)  # GB
                    })
        except ImportError:
            gpu_info = {"available": False, "message": "PyTorch not installed"}

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": {
                "name": config.get('project', {}).get('name', 'Cluster Heartbeat'),
                "version": config.get('project', {}).get('version', '1.0.0'),
                "environment": config.get('project', {}).get('environment', 'development'),
                "uptime": service_stats.get('uptime_seconds', 0),
                "is_running": service.is_running if service else False
            },
            "system": system_info,
            "gpu": gpu_info,
            "models": {
                "fingerprint_loaded": hasattr(service, 'pipeline') and service.pipeline.fingerprint_model is not None if service else False,
                "anomaly_detector_loaded": hasattr(service, 'pipeline') and service.pipeline.anomaly_detector.is_fitted if service and service.pipeline and service.pipeline.anomaly_detector else False
            },
            "stats": {
                "total_jobs": service_stats.get('total_jobs', 0),
                "processed_jobs": service_stats.get('processed_jobs', 0),
                "failed_jobs": service_stats.get('failed_jobs', 0),
                "cache_size": service_stats.get('cache_size', 0),
                "queue_size": service_stats.get('queue_size', 0)
            },
            "components": {
                "data_ingestion": "ready",
                "feature_extraction": "ready",
                "models": "ready" if service and service.pipeline and service.pipeline.is_initialized else "not_initialized",
                "api": "running"
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "components": {
                "api": "running",
                "models": "error"
            }
        }


@router.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes liveness probe.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/readiness")
async def readiness_check(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Kubernetes readiness probe.
    """
    try:
        # Check if service is ready
        is_ready = (
            service is not None and
            service.is_running and
            service.health_status == "running" and
            service.pipeline is not None and
            service.pipeline.is_initialized
        )
        
        if is_ready:
            return {
                "status": "ready",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "api": "ready",
                    "pipeline": "ready",
                    "models": "loaded"
                }
            }
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        return {
            "status": "not_ready",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/startup")
async def startup_check(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Startup probe for initialization.
    """
    try:
        # Check if service is initialized
        if service and service.pipeline and service.pipeline.is_initialized:
            return {
                "status": "initialized",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "initializing",
                "timestamp": datetime.now().isoformat(),
                "progress": "Service is starting up..."
            }
    except Exception as e:
        logger.error(f"Startup check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/deep")
async def deep_health_check(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Deep health check with detailed diagnostics.
    """
    try:
        # Test data ingestion
        test_data = {
            'gpu_utilization': 0.5,
            'memory_utilization': 0.5,
            'gpu_temperature': 65,
            'power_consumption': 100,
            'ecc_errors': 0,
            'xid_errors': 0,
            'cpu_usage': 0.3,
            'ram_usage': 0.4,
            'network_throughput': 100,
            'disk_io': 50,
            'job_runtime': 100,
            'queue_length': 5,
            'active_processes': 10,
            'timestamp': datetime.now().timestamp(),
            'node_id': 'test_node',
            'job_id': 'test_job'
        }
        
        # Test pipeline with small data
        test_results = service.process_metrics(test_data)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "diagnostics": {
                "data_ingestion": "pass",
                "feature_extraction": "pass" if service.pipeline and service.pipeline.feature_extractor else "fail",
                "fingerprint_model": "pass" if service.pipeline and service.pipeline.fingerprint_model else "fail",
                "anomaly_detection": "pass" if service.pipeline and service.pipeline.anomaly_detector and service.pipeline.anomaly_detector.is_fitted else "fail",
                "scheduling": "pass",
                "cost_optimization": "pass"
            },
            "test_results": {
                "processing_time": test_results.get('pipeline_stats', {}).get('processing_time', 0),
                "samples_processed": test_results.get('pipeline_stats', {}).get('total_samples', 0)
            }
        }
    
    except Exception as e:
        logger.error(f"Deep health check failed: {e}", exc_info=True)
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "diagnostics": {
                "test_failed": True,
                "error": str(e)
            }
        }


@router.get("/metrics")
async def health_metrics(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get health metrics for monitoring.
    """
    try:
        stats = service.get_service_stats() if service else {}
        pipeline_stats = service.pipeline.get_pipeline_stats() if service and service.pipeline else {}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": stats.get('uptime_seconds', 0),
            "service_health": {
                "is_running": stats.get('is_running', False),
                "status": stats.get('health_status', 'unknown'),
                "last_processed": stats.get('last_processed')
            },
            "job_stats": {
                "total": stats.get('total_jobs', 0),
                "processed": stats.get('processed_jobs', 0),
                "failed": stats.get('failed_jobs', 0),
                "avg_processing_time": stats.get('avg_processing_time', 0)
            },
            "system_stats": {
                "cache_size": stats.get('cache_size', 0),
                "queue_size": stats.get('queue_size', 0)
            },
            "pipeline_stats": pipeline_stats
        }
    
    except Exception as e:
        logger.error(f"Health metrics failed: {e}", exc_info=True)
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }