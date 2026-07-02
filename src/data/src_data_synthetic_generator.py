"""
Synthetic data generator for Cluster Heartbeat.
Generates realistic GPU cluster telemetry data with patterns and anomalies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from scipy import signal
from scipy.stats import expon, norm

from .ingestion import ClusterMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SyntheticPattern:
    """Container for synthetic data patterns."""
    name: str
    amplitude: float
    frequency: float
    phase: float
    offset: float
    noise_std: float
    anomaly_probability: float


class SyntheticDataGenerator:
    """
    Generates synthetic GPU cluster metrics with realistic patterns.
    Simulates normal operation, anomalies, and failures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize synthetic data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.synthetic_config = config['data']['synthetic']
        self.metrics = config['features']['metrics']
        self.window_size = config['data']['processing']['window_size']
        
        # Parameters
        self.num_nodes = self.synthetic_config.get('num_nodes', 10)
        self.num_jobs = self.synthetic_config.get('num_jobs', 100)
        self.time_steps = self.synthetic_config.get('time_steps', 1000)
        self.seed = self.synthetic_config.get('seed', 42)
        
        # Set random seed
        np.random.seed(self.seed)
        
        # Patterns for different metrics
        self.patterns = self._initialize_patterns()
        
        # State
        self.node_ids = None
        self.job_ids = None
        self.timestamps = None
        
        logger.info(f"SyntheticDataGenerator initialized with {self.num_nodes} nodes, {self.num_jobs} jobs, {self.time_steps} time steps")
    
    def _initialize_patterns(self) -> Dict[str, SyntheticPattern]:
        """
        Initialize patterns for each metric.
        
        Returns:
            Dictionary mapping metric names to patterns
        """
        patterns = {}
        
        for metric in self.metrics:
            if metric in ['gpu_utilization', 'cpu_usage']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=0.4,
                    frequency=0.02,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=0.3,
                    noise_std=0.05,
                    anomaly_probability=0.02
                )
            elif metric in ['memory_utilization', 'ram_usage']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=0.3,
                    frequency=0.01,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=0.4,
                    noise_std=0.04,
                    anomaly_probability=0.01
                )
            elif metric in ['gpu_temperature']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=15.0,
                    frequency=0.015,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=50.0,
                    noise_std=2.0,
                    anomaly_probability=0.015
                )
            elif metric in ['power_consumption']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=50.0,
                    frequency=0.02,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=100.0,
                    noise_std=10.0,
                    anomaly_probability=0.01
                )
            elif metric in ['ecc_errors', 'xid_errors']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=0.5,
                    frequency=0.001,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=0.1,
                    noise_std=0.2,
                    anomaly_probability=0.005
                )
            elif metric in ['network_throughput']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=200.0,
                    frequency=0.025,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=300.0,
                    noise_std=30.0,
                    anomaly_probability=0.01
                )
            elif metric in ['disk_io']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=50.0,
                    frequency=0.02,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=100.0,
                    noise_std=15.0,
                    anomaly_probability=0.01
                )
            elif metric in ['job_runtime']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=50.0,
                    frequency=0.001,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=100.0,
                    noise_std=20.0,
                    anomaly_probability=0.005
                )
            elif metric in ['queue_length']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=5.0,
                    frequency=0.015,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=10.0,
                    noise_std=2.0,
                    anomaly_probability=0.01
                )
            elif metric in ['active_processes']:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=15.0,
                    frequency=0.02,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=25.0,
                    noise_std=3.0,
                    anomaly_probability=0.01
                )
            else:
                patterns[metric] = SyntheticPattern(
                    name=metric,
                    amplitude=0.5,
                    frequency=0.01,
                    phase=np.random.uniform(0, 2*np.pi),
                    offset=0.5,
                    noise_std=0.1,
                    anomaly_probability=0.01
                )
        
        return patterns
    
    def generate(self) -> ClusterMetrics:
        """
        Generate synthetic cluster metrics.
        
        Returns:
            ClusterMetrics object with generated data
        """
        # Generate base time series
        t = np.arange(self.time_steps)
        self.timestamps = t
        
        # Generate node and job IDs
        self.node_ids = np.random.choice(range(self.num_nodes), self.time_steps)
        self.job_ids = np.random.choice(range(self.num_jobs), self.time_steps)
        
        # Generate metrics
        metrics_data = {}
        for metric in self.metrics:
            if metric in self.patterns:
                metrics_data[metric] = self._generate_metric(metric, t)
            else:
                logger.warning(f"Unknown metric: {metric}, generating default")
                metrics_data[metric] = self._generate_default_metric(t)
        
        # Add anomalies and failure patterns
        self._add_anomalies(metrics_data, t)
        self._add_failure_patterns(metrics_data, t)
        
        # Prepare arrays
        arrays = {}
        for metric in self.metrics:
            if metric in metrics_data:
                arrays[metric] = np.array(metrics_data[metric]).reshape(-1, 1)
            else:
                arrays[metric] = np.zeros((self.time_steps, 1))
        
        # Ensure all metrics have the correct shape
        for metric in self.metrics:
            if metric not in arrays:
                arrays[metric] = np.zeros((self.time_steps, 1))
        
        # Create timestamps (current time + time steps)
        start_time = datetime.now()
        timestamps = np.array([(start_time + timedelta(seconds=i)).timestamp() for i in range(self.time_steps)]).reshape(-1, 1)
        
        # Return ClusterMetrics
        return ClusterMetrics(
            gpu_utilization=arrays.get('gpu_utilization', np.zeros((self.time_steps, 1))),
            memory_utilization=arrays.get('memory_utilization', np.zeros((self.time_steps, 1))),
            gpu_temperature=arrays.get('gpu_temperature', np.zeros((self.time_steps, 1))),
            power_consumption=arrays.get('power_consumption', np.zeros((self.time_steps, 1))),
            ecc_errors=arrays.get('ecc_errors', np.zeros((self.time_steps, 1))),
            xid_errors=arrays.get('xid_errors', np.zeros((self.time_steps, 1))),
            cpu_usage=arrays.get('cpu_usage', np.zeros((self.time_steps, 1))),
            ram_usage=arrays.get('ram_usage', np.zeros((self.time_steps, 1))),
            network_throughput=arrays.get('network_throughput', np.zeros((self.time_steps, 1))),
            disk_io=arrays.get('disk_io', np.zeros((self.time_steps, 1))),
            job_runtime=arrays.get('job_runtime', np.zeros((self.time_steps, 1))),
            queue_length=arrays.get('queue_length', np.zeros((self.time_steps, 1))),
            active_processes=arrays.get('active_processes', np.zeros((self.time_steps, 1))),
            timestamp=timestamps,
            node_ids=self.node_ids.reshape(-1, 1),
            job_ids=self.job_ids.reshape(-1, 1)
        )
    
    def _generate_metric(self, metric: str, t: np.ndarray) -> np.ndarray:
        """
        Generate a specific metric with its pattern.
        
        Args:
            metric: Metric name
            t: Time array
            
        Returns:
            Generated metric values
        """
        pattern = self.patterns[metric]
        
        # Generate base signal
        base_signal = (
            pattern.offset +
            pattern.amplitude * np.sin(2 * np.pi * pattern.frequency * t + pattern.phase)
        )
        
        # Add noise
        noise = np.random.normal(0, pattern.noise_std, len(t))
        
        # Combine
        signal_values = base_signal + noise
        
        # Apply metric-specific constraints
        if 'utilization' in metric or 'usage' in metric:
            signal_values = np.clip(signal_values, 0, 1)
        elif 'temperature' in metric:
            signal_values = np.clip(signal_values, 20, 100)
        elif 'power' in metric:
            signal_values = np.maximum(signal_values, 0)
        elif 'errors' in metric:
            signal_values = np.maximum(signal_values, 0)
        
        return signal_values
    
    def _generate_default_metric(self, t: np.ndarray) -> np.ndarray:
        """
        Generate a default metric.
        
        Args:
            t: Time array
            
        Returns:
            Generated metric values
        """
        return 0.5 + 0.3 * np.sin(0.02 * t) + 0.1 * np.random.randn(len(t))
    
    def _add_anomalies(self, metrics_data: Dict[str, np.ndarray], t: np.ndarray) -> None:
        """
        Add anomalies to the generated metrics.
        
        Args:
            metrics_data: Dictionary of metrics
            t: Time array
        """
        # Add spikes
        anomaly_indices = np.random.choice(
            range(self.time_steps),
            size=int(0.02 * self.time_steps),
            replace=False
        )
        
        for metric, data in metrics_data.items():
            pattern = self.patterns.get(metric)
            if pattern is None:
                continue
            
            # Anomaly probability
            for idx in anomaly_indices:
                if np.random.random() < pattern.anomaly_probability:
                    # Add spike
                    spike = np.random.uniform(2, 5) * pattern.noise_std
                    data[idx] += spike
        
        # Add gradual degradation for some metrics
        degradation_start = int(self.time_steps * 0.7)
        degradation_end = self.time_steps
        
        for metric, data in metrics_data.items():
            if 'utilization' in metric or 'temperature' in metric:
                # Add gradual degradation for some nodes
                if np.random.random() < 0.2:  # 20% chance
                    progress = np.linspace(0, 1, degradation_end - degradation_start)
                    if 'utilization' in metric:
                        data[degradation_start:degradation_end] *= (1 - 0.3 * progress)
                    elif 'temperature' in metric:
                        data[degradation_start:degradation_end] += 10 * progress
    
    def _add_failure_patterns(self, metrics_data: Dict[str, np.ndarray], t: np.ndarray) -> None:
        """
        Add failure patterns to the generated metrics.
        
        Args:
            metrics_data: Dictionary of metrics
            t: Time array
        """
        # Simulate GPU failures
        num_failures = np.random.poisson(0.5)  # Average 0.5 failures per run
        
        for _ in range(num_failures):
            # Random failure start
            failure_start = np.random.randint(int(0.5 * self.time_steps), int(0.9 * self.time_steps))
            failure_duration = np.random.randint(50, 200)
            failure_end = min(failure_start + failure_duration, self.time_steps)
            
            # Affected metrics
            affected_metrics = np.random.choice(
                ['gpu_utilization', 'gpu_temperature', 'ecc_errors', 'xid_errors'],
                size=np.random.randint(1, 3),
                replace=False
            )
            
            for metric in affected_metrics:
                if metric in metrics_data:
                    # Gradually degrade
                    progress = np.linspace(0, 1, failure_end - failure_start)
                    if 'utilization' in metric:
                        metrics_data[metric][failure_start:failure_end] *= (1 - 0.5 * progress)
                    elif 'temperature' in metric:
                        metrics_data[metric][failure_start:failure_end] += 15 * progress
                    elif 'errors' in metric:
                        metrics_data[metric][failure_start:failure_end] += 0.5 + 2 * progress
    
    def generate_node_data(self, node_id: int, time_steps: Optional[int] = None) -> pd.DataFrame:
        """
        Generate data for a specific node.
        
        Args:
            node_id: Node identifier
            time_steps: Number of time steps (uses default if None)
            
        Returns:
            DataFrame with node data
        """
        if time_steps is None:
            time_steps = self.time_steps
        
        # Generate data for this node
        metrics_data = {}
        for metric in self.metrics:
            t = np.arange(time_steps)
            if metric in self.patterns:
                metrics_data[metric] = self._generate_metric(metric, t)
            else:
                metrics_data[metric] = self._generate_default_metric(t)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        df['node_id'] = node_id
        df['job_id'] = np.random.choice(range(self.num_jobs), time_steps)
        df['timestamp'] = np.arange(time_steps)
        
        return df
    
    def get_metric_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics of generated metrics.
        
        Returns:
            Dictionary with metric statistics
        """
        # Generate sample data
        metrics_data = {}
        t = np.arange(self.time_steps)
        
        for metric in self.metrics:
            if metric in self.patterns:
                data = self._generate_metric(metric, t)
            else:
                data = self._generate_default_metric(t)
            metrics_data[metric] = data
        
        # Calculate statistics
        stats = {}
        for metric, data in metrics_data.items():
            stats[metric] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75))
            }
        
        return stats
    
    def generate_batch(self, batch_size: int = 100, num_batches: int = 10) -> List[pd.DataFrame]:
        """
        Generate multiple batches of data.
        
        Args:
            batch_size: Size of each batch
            num_batches: Number of batches
            
        Returns:
            List of DataFrames
        """
        batches = []
        
        for _ in range(num_batches):
            # Generate data
            metrics_data = {}
            t = np.arange(batch_size)
            
            for metric in self.metrics:
                if metric in self.patterns:
                    metrics_data[metric] = self._generate_metric(metric, t)
                else:
                    metrics_data[metric] = self._generate_default_metric(t)
            
            # Create DataFrame
            df = pd.DataFrame(metrics_data)
            df['node_id'] = np.random.choice(range(self.num_nodes), batch_size)
            df['job_id'] = np.random.choice(range(self.num_jobs), batch_size)
            df['timestamp'] = np.arange(batch_size)
            
            batches.append(df)
        
        return batches


# Convenience function for quick generation
def generate_synthetic_data(config: Dict[str, Any]) -> ClusterMetrics:
    """
    Quick function to generate synthetic data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ClusterMetrics object
    """
    generator = SyntheticDataGenerator(config)
    return generator.generate()