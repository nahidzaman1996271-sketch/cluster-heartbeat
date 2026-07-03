"""
Test suite for Cluster Heartbeat.
Contains unit tests for all modules.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_CONFIG = {
    'data_dir': Path(__file__).parent / 'test_data',
    'model_dir': Path(__file__).parent / 'test_models',
    'log_level': 'DEBUG'
}

# Create test directories
TEST_CONFIG['data_dir'].mkdir(exist_ok=True)
TEST_CONFIG['model_dir'].mkdir(exist_ok=True)

__all__ = ['TEST_CONFIG']