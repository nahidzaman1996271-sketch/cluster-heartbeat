"""
Cluster Heartbeat - AI-powered GPU cluster intelligence system
One Signal, Three Outcomes

Package: src
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Slow Walker Team"
__description__ = "AI-powered GPU cluster intelligence system"

# Import main components for easy access
from .core.pipeline import ClusterHeartbeatPipeline
from .core.service import ClusterHeartbeatService
from .config import load_config

# Define what's available when someone imports the package
__all__ = [
    'ClusterHeartbeatPipeline',
    'ClusterHeartbeatService',
    'load_config',
    '__version__',
    '__author__',
    '__description__'
]

# Package metadata
PACKAGE_NAME = "cluster-heartbeat"
PACKAGE_VERSION = __version__