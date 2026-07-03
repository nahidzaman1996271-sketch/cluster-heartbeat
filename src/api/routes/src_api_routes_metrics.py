"""
Metrics routes for Cluster Heartbeat API.
Provides endpoints for ingesting and retrieving metrics.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from ..dependencies import get_service, rate_limit, validate_metrics_data
from ..schemas.models import MetricsIngestRequest, MetricsBatchIngestRequest
from ...core.service import ClusterHeartbeatService
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/ingest")
async def ingest_metrics(
    request: MetricsIngestRequest,
    background_tasks: BackgroundTasks,
    service: ClusterHeartbeatService = Depends(get_service),
    _: None = Depends(rate_limit)
) -> Dict[str, Any]:
    """
    Ingest a single metrics record for processing.
    
    Args:
        request: Metrics data
        background_tasks: FastAPI background tasks
        
    Returns:
        Job ID for tracking
    """
    try:
        # Convert to dict
        metrics_data = request.dict()
        
        # Process asynchronously
        job_id = service.process_async(metrics_data)
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Metrics accepted for processing"
        }
    
    except Exception as e:
        logger.error(f"Metrics ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/batch")
async def ingest_batch_metrics(
    request: MetricsBatchIngestRequest,
    background_tasks: BackgroundTasks,
    service: ClusterHeartbeatService = Depends(get_service),
    _: None = Depends(rate_limit)
) -> Dict[str, Any]:
    """
    Ingest a batch of metrics records for processing.
    
    Args:
        request: Batch metrics data
        background_tasks: FastAPI background tasks
        
    Returns:
        Job ID for tracking
    """
    try:
        # Convert to list of dicts
        metrics_data = [item.dict() for item in request.metrics]
        
        # Process asynchronously
        job_id = service.process_async(metrics_data)
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "count": len(metrics_data),
            "timestamp": datetime.now().isoformat(),
            "message": f"Batch of {len(metrics_data)} metrics accepted for processing"
        }
    
    except Exception as e:
        logger.error(f"Batch metrics ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}")
async def get_processing_status(
    job_id: str,
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get processing status for a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Processing status
    """
    try:
        status = service.get_job_status(job_id)
        
        if status.get('status') == 'not_found':
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return {
            "job_id": job_id,
            "status": status.get('status', 'unknown'),
            "timestamp": status.get('timestamp'),
            "has_results": status.get('has_results', False)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get processing status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{job_id}")
async def get_processing_results(
    job_id: str,
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get processing results for a job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Processing results
    """
    try:
        results = service.get_job_results(job_id)
        
        if 'error' in results and results.get('error') == 'Results not found':
            # Check if job is still processing
            status = service.get_job_status(job_id)
            if status.get('status') == 'queued' or status.get('status') == 'processing':
                return {
                    "job_id": job_id,
                    "status": "processing",
                    "message": "Job is still processing",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=404, detail=f"Results for job {job_id} not found")
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get processing results error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cluster")
async def get_cluster_metrics(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get current cluster metrics summary.
    
    Returns:
        Cluster metrics summary
    """
    try:
        status = service.get_cluster_status()
        
        return {
            "metrics": status,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Get cluster metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/{node_id}")
async def get_node_metrics(
    node_id: str,
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get metrics for a specific node.
    
    Args:
        node_id: Node identifier
        
    Returns:
        Node metrics
    """
    try:
        # Get latest results
        if 'last_results' not in service.cache:
            raise HTTPException(status_code=404, detail="No data available")
        
        results = service.cache['last_results']
        
        # Extract node-specific data
        health_scores = results.get('health_scores', {})
        node_score = health_scores.get('node_scores', {}).get(node_id, None)
        
        if node_score is None:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        
        return {
            "node_id": node_id,
            "health_score": node_score,
            "status": "healthy" if node_score > 70 else "warning" if node_score > 50 else "critical",
            "timestamp": results.get('timestamp')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get node metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes")
async def get_all_nodes(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get metrics for all nodes.
    
    Returns:
        All nodes metrics
    """
    try:
        # Get latest results
        if 'last_results' not in service.cache:
            raise HTTPException(status_code=404, detail="No data available")
        
        results = service.cache['last_results']
        health_scores = results.get('health_scores', {})
        
        nodes = []
        for node_id, score in health_scores.get('node_scores', {}).items():
            nodes.append({
                "node_id": node_id,
                "health_score": score,
                "status": "healthy" if score > 70 else "warning" if score > 50 else "critical"
            })
        
        return {
            "nodes": nodes,
            "total": len(nodes),
            "timestamp": results.get('timestamp')
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get all nodes metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
async def get_latest_metrics(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get latest processed metrics.
    
    Returns:
        Latest metrics
    """
    try:
        if 'last_results' not in service.cache:
            raise HTTPException(status_code=404, detail="No data available")
        
        return service.cache['last_results']
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get latest metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{node_id}")
async def get_node_history(
    node_id: str,
    hours: int = Query(24, ge=1, le=168),
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get historical metrics for a node.
    
    Args:
        node_id: Node identifier
        hours: Number of hours of history
        
    Returns:
        Historical metrics
    """
    try:
        # In production, this would query a time-series database
        # For now, return synthetic history
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)]
        
        return {
            "node_id": node_id,
            "history": {
                "timestamps": [ts.isoformat() for ts in reversed(timestamps)],
                "health_scores": [70 + 20 * np.sin(i/10) + 10 * np.random.randn() for i in range(hours)],
                "gpu_utilization": [0.5 + 0.4 * np.sin(i/5) + 0.1 * np.random.randn() for i in range(hours)],
                "memory_utilization": [0.4 + 0.3 * np.cos(i/7) + 0.05 * np.random.randn() for i in range(hours)]
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Get node history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Import numpy for history generation
import numpy as np