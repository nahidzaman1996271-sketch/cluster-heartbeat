"""
API Routes module for Cluster Heartbeat.
Contains all route handlers for different API endpoints.
"""

from .health import router as health_router
from .metrics import router as metrics_router
from .predictions import router as predictions_router
from .recommendations import router as recommendations_router

__all__ = [
    'health_router',
    'metrics_router',
    'predictions_router',
    'recommendations_router'
]