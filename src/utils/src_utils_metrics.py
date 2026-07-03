"""
Metrics collection utilities for Cluster Heartbeat.
Tracks and aggregates system and application metrics.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime
import threading
import psutil
import json
import os

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class MetricsCollector:
    """
    Collects and aggregates system metrics.
    Supports real-time metric tracking and aggregation.
    """
    
    def __init__(self, max_history: int = 1000, enable_stats: bool = True):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of points to keep in history
            enable_stats: Whether to compute statistics
        """
        self.max_history = max_history
        self.enable_stats = enable_stats
        
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.history: deque = deque(maxlen=max_history)
        self.tags: Dict[str, str] = {}
        self._lock = threading.Lock()
        
        # Statistics
        self.stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"MetricsCollector initialized with max_history={max_history}")
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        with self._lock:
            if tags is None:
                tags = {}
            
            metric = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags
            )
            
            # Store in history
            self.history.append(metric)
            
            # Store in metrics dict
            if name not in self.metrics:
                self.metrics[name] = {
                    'values': deque(maxlen=self.max_history),
                    'timestamps': deque(maxlen=self.max_history)
                }
            
            self.metrics[name]['values'].append(value)
            self.metrics[name]['timestamps'].append(metric.timestamp)
            
            # Update tags
            if tags:
                self.tags.update(tags)
    
    def record_batch(self, metrics: List[Dict[str, Any]]) -> None:
        """
        Record multiple metrics.
        
        Args:
            metrics: List of metric dictionaries
        """
        for metric in metrics:
            self.record(
                name=metric.get('name', 'unknown'),
                value=metric.get('value', 0.0),
                tags=metric.get('tags', {})
            )
    
    def get_metric_stats(self, name: str) -> Dict[str, Any]:
        """
        Get statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Statistics dictionary
        """
        with self._lock:
            if name not in self.metrics:
                return {}
            
            values = list(self.metrics[name]['values'])
            
            if not values:
                return {}
            
            values_array = np.array(values)
            
            stats = {
                'name': name,
                'count': len(values),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'q25': float(np.percentile(values_array, 25)),
                'q75': float(np.percentile(values_array, 75)),
                'sum': float(np.sum(values_array)),
                'trend': self._calculate_trend(values_array),
                'recent_value': float(values[-1]) if values else 0
            }
            
            return stats
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """
        Calculate trend direction (-1 to 1).
        
        Args:
            values: Array of values
            
        Returns:
            Trend value between -1 and 1
        """
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize trend
        std_val = np.std(values) if np.std(values) > 0 else 1
        trend = np.clip(slope / (std_val + 1e-8), -1, 1)
        
        return float(trend)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all tracked metrics with statistics.
        
        Returns:
            All metrics with statistics
        """
        with self._lock:
            metrics_stats = {}
            for name in self.metrics:
                stats = self.get_metric_stats(name)
                if stats:
                    metrics_stats[name] = stats
            
            return {
                'metrics': metrics_stats,
                'total_points': len(self.history),
                'timestamp': time.time(),
                'tags': self.tags
            }
    
    def get_latest(self, name: str) -> Optional[float]:
        """
        Get the latest value for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Latest value or None
        """
        with self._lock:
            if name in self.metrics and self.metrics[name]['values']:
                return float(self.metrics[name]['values'][-1])
            return None
    
    def get_history(self, name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of a metric.
        
        Args:
            name: Metric name
            limit: Optional limit on number of points
            
        Returns:
            List of metric points
        """
        with self._lock:
            if name not in self.metrics:
                return []
            
            timestamps = list(self.metrics[name]['timestamps'])
            values = list(self.metrics[name]['values'])
            
            if limit and limit < len(values):
                timestamps = timestamps[-limit:]
                values = values[-limit:]
            
            return [
                {'timestamp': t, 'value': v}
                for t, v in zip(timestamps, values)
            ]
    
    def get_aggregate(self, name: str, aggregation: str = 'mean') -> float:
        """
        Get aggregate value for a metric.
        
        Args:
            name: Metric name
            aggregation: Aggregation type (mean, sum, min, max, count)
            
        Returns:
            Aggregate value
        """
        stats = self.get_metric_stats(name)
        if not stats:
            return 0.0
        
        if aggregation == 'mean':
            return stats.get('mean', 0.0)
        elif aggregation == 'sum':
            return stats.get('sum', 0.0)
        elif aggregation == 'min':
            return stats.get('min', 0.0)
        elif aggregation == 'max':
            return stats.get('max', 0.0)
        elif aggregation == 'count':
            return float(stats.get('count', 0))
        else:
            return stats.get(aggregation, 0.0)
    
    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self.metrics.clear()
            self.history.clear()
            self.stats.clear()
            logger.info("Metrics cleared")
    
    def export(self) -> Dict[str, Any]:
        """
        Export all metrics as dictionary.
        
        Returns:
            Metrics data
        """
        with self._lock:
            return {
                'metrics': {
                    name: {
                        'values': list(self.metrics[name]['values']),
                        'timestamps': list(self.metrics[name]['timestamps'])
                    }
                    for name in self.metrics
                },
                'history': [m.to_dict() for m in self.history],
                'stats': self.stats,
                'tags': self.tags,
                'timestamp': time.time()
            }
    
    def import_data(self, data: Dict[str, Any]) -> None:
        """
        Import metrics data.
        
        Args:
            data: Metrics data dictionary
        """
        with self._lock:
            if 'metrics' in data:
                for name, metric_data in data['metrics'].items():
                    if name not in self.metrics:
                        self.metrics[name] = {
                            'values': deque(maxlen=self.max_history),
                            'timestamps': deque(maxlen=self.max_history)
                        }
                    
                    for value, timestamp in zip(
                        metric_data.get('values', []),
                        metric_data.get('timestamps', [])
                    ):
                        self.metrics[name]['values'].append(value)
                        self.metrics[name]['timestamps'].append(timestamp)
            
            if 'history' in data:
                for point in data['history']:
                    self.history.append(MetricPoint(
                        name=point.get('name', 'unknown'),
                        value=point.get('value', 0.0),
                        timestamp=point.get('timestamp', time.time()),
                        tags=point.get('tags', {})
                    ))


class SystemMetricsCollector(MetricsCollector):
    """
    Collects system-level metrics (CPU, memory, disk, network).
    """
    
    def __init__(self, interval: int = 60, **kwargs):
        """
        Initialize system metrics collector.
        
        Args:
            interval: Collection interval in seconds
            **kwargs: Additional arguments for MetricsCollector
        """
        super().__init__(**kwargs)
        self.interval = interval
        self._running = False
        self._thread = None
        
        logger.info(f"SystemMetricsCollector initialized with interval={interval}s")
    
    def start(self) -> None:
        """Start collecting system metrics."""
        if self._running:
            logger.warning("Collection already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info("System metrics collection started")
    
    def stop(self) -> None:
        """Stop collecting system metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("System metrics collection stopped")
    
    def _collect_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                self.collect()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            # Sleep for interval
            for _ in range(self.interval):
                if not self._running:
                    break
                time.sleep(1)
    
    def collect(self) -> Dict[str, Any]:
        """
        Collect system metrics.
        
        Returns:
            Collected metrics
        """
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_avg = np.mean(cpu_percent)
        
        self.record('cpu.percent', cpu_avg)
        for i, percent in enumerate(cpu_percent):
            self.record(f'cpu.core_{i}.percent', percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.record('memory.percent', memory.percent)
        self.record('memory.used_gb', memory.used / (1024**3))
        self.record('memory.available_gb', memory.available / (1024**3))
        self.record('memory.total_gb', memory.total / (1024**3))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.record('disk.percent', disk.percent)
        self.record('disk.used_gb', disk.used / (1024**3))
        self.record('disk.free_gb', disk.free / (1024**3))
        
        # Network metrics (if available)
        try:
            net = psutil.net_io_counters()
            self.record('network.sent_mb', net.bytes_sent / (1024**2))
            self.record('network.recv_mb', net.bytes_recv / (1024**2))
            self.record('network.packets_sent', net.packets_sent)
            self.record('network.packets_recv', net.packets_recv)
        except Exception:
            pass
        
        # Process metrics
        self.record('process.count', len(psutil.pids()))
        
        # GPU metrics (if available)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    self.record(f'gpu.{i}.memory_allocated_gb', 
                               torch.cuda.memory_allocated(i) / (1024**3))
                    self.record(f'gpu.{i}.memory_reserved_gb',
                               torch.cuda.memory_reserved(i) / (1024**3))
        except ImportError:
            pass
        
        return self.get_all_metrics()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            System status
        """
        return {
            'cpu_percent': self.get_latest('cpu.percent'),
            'memory_percent': self.get_latest('memory.percent'),
            'memory_available_gb': self.get_latest('memory.available_gb'),
            'disk_percent': self.get_latest('disk.percent'),
            'process_count': self.get_latest('process.count'),
            'timestamp': time.time()
        }


# Convenience function
def collect_system_metrics() -> Dict[str, Any]:
    """
    Collect system metrics once.
    
    Returns:
        System metrics
    """
    collector = SystemMetricsCollector()
    return collector.collect()