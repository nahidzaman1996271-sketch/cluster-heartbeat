"""
FastAPI application for Cluster Heartbeat.
Provides REST API endpoints for all cluster intelligence services.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio
import os
from pathlib import Path

from .routes import health, metrics, predictions, recommendations
from .dependencies import get_service, get_config
from ..utils.logger import get_logger
from ..core.service import ClusterHeartbeatService

logger = get_logger(__name__)

# Load configuration
config = get_config()

# Initialize FastAPI app
app = FastAPI(
    title=config.get('api', {}).get('docs', {}).get('title', 'Cluster Heartbeat API'),
    description=config.get('api', {}).get('docs', {}).get('description', 
        'AI-powered GPU cluster intelligence system - One Signal, Three Outcomes'),
    version=config.get('api', {}).get('docs', {}).get('version', '1.0.0'),
    openapi_url=config.get('api', {}).get('docs', {}).get('openapi_url', '/openapi.json'),
    docs_url=config.get('api', {}).get('docs', {}).get('docs_url', '/docs'),
    redoc_url=config.get('api', {}).get('docs', {}).get('redoc_url', '/redoc'),
    contact={
        "name": "Slow Walker Team",
        "email": "your-email@example.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Configure CORS
cors_config = config.get('api', {}).get('cors', {})
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config.get('allowed_origins', ["*"]),
    allow_credentials=True,
    allow_methods=cors_config.get('allowed_methods', ["*"]),
    allow_headers=cors_config.get('allowed_headers', ["*"]),
    max_age=cors_config.get('max_age', 3600)
)

# Include routers
app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["Health"]
)
app.include_router(
    metrics.router,
    prefix="/api/v1/metrics",
    tags=["Metrics"]
)
app.include_router(
    predictions.router,
    prefix="/api/v1/predictions",
    tags=["Predictions"]
)
app.include_router(
    recommendations.router,
    prefix="/api/v1/recommendations",
    tags=["Recommendations"]
)


# ============================================
# Root Endpoints
# ============================================

@app.get("/")
async def root():
    """
    Root endpoint with service information.
    """
    return {
        "service": "Cluster Heartbeat",
        "version": config.get('project', {}).get('version', '1.0.0'),
        "description": "AI-powered GPU cluster intelligence system",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics",
            "predictions": "/api/v1/predictions",
            "recommendations": "/api/v1/recommendations",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/ping")
async def ping():
    """
    Simple ping endpoint for connectivity testing.
    """
    return {
        "pong": True,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/version")
async def version():
    """
    Get service version information.
    """
    return {
        "service": "Cluster Heartbeat",
        "version": config.get('project', {}).get('version', '1.0.0'),
        "environment": config.get('project', {}).get('environment', 'development'),
        "api_version": "v1",
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# Dashboard Endpoints
# ============================================

@app.get("/dashboard")
async def get_dashboard(
    service: ClusterHeartbeatService = Depends(get_service)
):
    """
    Get complete dashboard data.
    
    Returns:
        Complete dashboard data JSON
    """
    try:
        return service.get_dashboard_data()
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/summary")
async def get_dashboard_summary(
    service: ClusterHeartbeatService = Depends(get_service)
):
    """
    Get dashboard summary only.
    
    Returns:
        Dashboard summary
    """
    try:
        data = service.get_dashboard_data()
        return {
            "cluster_summary": data.get("cluster_summary", {}),
            "health_scores": data.get("health_scores", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Dashboard summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Service Management Endpoints
# ============================================

@app.get("/service/status")
async def get_service_status(
    service: ClusterHeartbeatService = Depends(get_service)
):
    """
    Get service status.
    
    Returns:
        Service status
    """
    try:
        return {
            "status": "running",
            "uptime_seconds": service.get_service_stats().get('uptime_seconds', 0),
            "health": service.health_status,
            "is_running": service.is_running,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Service status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/service/stats")
async def get_service_stats(
    service: ClusterHeartbeatService = Depends(get_service)
):
    """
    Get service statistics.
    
    Returns:
        Service statistics
    """
    try:
        return service.get_service_stats()
    except Exception as e:
        logger.error(f"Service stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/service/clear-cache")
async def clear_service_cache(
    service: ClusterHeartbeatService = Depends(get_service)
):
    """
    Clear service cache.
    
    Returns:
        Operation result
    """
    try:
        service.clear_cache()
        return {
            "status": "success",
            "message": "Cache cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Clear cache error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Metrics Endpoints (Direct)
# ============================================

@app.post("/metrics/ingest")
async def ingest_metrics(
    background_tasks: BackgroundTasks,
    metrics_data: Dict[str, Any],
    service: ClusterHeartbeatService = Depends(get_service)
):
    """
    Ingest metrics for processing.
    
    Args:
        metrics_data: Metrics data
        
    Returns:
        Job ID for tracking
    """
    try:
        # Process asynchronously
        job_id = service.process_async(metrics_data)
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/latest")
async def get_latest_metrics(
    service: ClusterHeartbeatService = Depends(get_service)
):
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


# ============================================
# Health Check Endpoints
# ============================================

@app.get("/healthz")
async def healthz():
    """
    Kubernetes-style health check.
    """
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(
    service: ClusterHeartbeatService = Depends(get_service)
):
    """
    Kubernetes-style readiness check.
    """
    if service.health_status == "running":
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/livez")
async def livez():
    """
    Kubernetes-style liveness check.
    """
    return {"status": "alive"}


# ============================================
# Static Files (Optional)
# ============================================

# Serve static files if directory exists
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom HTTP exception handler.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    General exception handler.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if config.get('project', {}).get('debug', False) else "An error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================
# Startup and Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    """
    logger.info("Starting Cluster Heartbeat API...")
    logger.info(f"Environment: {config.get('project', {}).get('environment', 'development')}")
    logger.info(f"Debug mode: {config.get('project', {}).get('debug', False)}")
    
    # Initialize service
    try:
        service = get_service()
        if not service.is_running:
            service.start()
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    """
    logger.info("Shutting down Cluster Heartbeat API...")
    
    # Shutdown service
    try:
        service = get_service()
        service.shutdown()
        logger.info("Service shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# ============================================
# Main Entry Point
# ============================================

def main():
    """
    Main entry point for running the API server.
    """
    api_config = config.get('api', {})
    
    uvicorn.run(
        "src.api.main:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        workers=api_config.get('workers', 4),
        reload=api_config.get('reload', False),
        log_level=config.get('logging', {}).get('level', 'info').lower(),
        access_log=True,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )


if __name__ == "__main__":
    main()