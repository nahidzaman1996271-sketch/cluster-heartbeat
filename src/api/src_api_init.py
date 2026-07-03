"""
API layer for Cluster Heartbeat.
Provides RESTful API endpoints for all cluster intelligence services.
"""

from .main import app
from .dependencies import get_service, get_config

__all__ = [
    'app',
    'get_service',
    'get_config'
]