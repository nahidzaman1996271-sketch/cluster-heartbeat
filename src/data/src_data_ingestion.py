"""
Data ingestion module for Cluster Heartbeat.
Handles loading data from various sources including synthetic generation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import csv
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterMetrics:
    """
    Container for cluster metrics data.
    """
    gpu_utilization: np.ndarray
    memory_utilization: np.ndarray
    gpu_temperature: np.ndarray
    power_consumption: np.ndarray
    ecc_errors: np.ndarray
    xid_errors: np.ndarray
    cpu_usage: np.ndarray
    ram_usage: np.ndarray
    network_throughput: np.ndarray
    disk_io: np.ndarray
    job_runtime: np.ndarray
    queue_length: np.ndarray
    active_processes: np.ndarray
    timestamp: np.ndarray
    node_ids: np.ndarray
    job_ids: np.ndarray
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.
        
        Returns:
            DataFrame with all metrics
        """
        return pd.DataFrame({
            'gpu_utilization': self.gpu_utilization.flatten(),
            'memory_utilization': self.memory_utilization.flatten(),
            'gpu_temperature': self.gpu_temperature.flatten(),
            'power_consumption': self.power_consumption.flatten(),
            'ecc_errors': self.ecc_errors.flatten(),
            'xid_errors': self.xid_errors.flatten(),
            'cpu_usage': self.cpu_usage.flatten(),
            'ram_usage': self.ram_usage.flatten(),
            'network_throughput': self.network_throughput.flatten(),
            'disk_io': self.disk_io.flatten(),
            'job_runtime': self.job_runtime.flatten(),
            'queue_length': self.queue_length.flatten(),
            'active_processes': self.active_processes.flatten(),
            'timestamp': self.timestamp.flatten(),
            'node_id': self.node_ids.flatten(),
            'job_id': self.job_ids.flatten()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'gpu_utilization': self.gpu_utilization.tolist(),
            'memory_utilization': self.memory_utilization.tolist(),
            'gpu_temperature': self.gpu_temperature.tolist(),
            'power_consumption': self.power_consumption.tolist(),
            'ecc_errors': self.ecc_errors.tolist(),
            'xid_errors': self.xid_errors.tolist(),
            'cpu_usage': self.cpu_usage.tolist(),
            'ram_usage': self.ram_usage.tolist(),
            'network_throughput': self.network_throughput.tolist(),
            'disk_io': self.disk_io.tolist(),
            'job_runtime': self.job_runtime.tolist(),
            'queue_length': self.queue_length.tolist(),
            'active_processes': self.active_processes.tolist(),
            'timestamp': self.timestamp.tolist(),
            'node_ids': self.node_ids.tolist(),
            'job_ids': self.job_ids.tolist()
        }
    
    def get_metrics_shape(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get shape of each metric array.
        
        Returns:
            Dictionary with metric names and their shapes
        """
        return {
            'gpu_utilization': self.gpu_utilization.shape,
            'memory_utilization': self.memory_utilization.shape,
            'gpu_temperature': self.gpu_temperature.shape,
            'power_consumption': self.power_consumption.shape,
            'ecc_errors': self.ecc_errors.shape,
            'xid_errors': self.xid_errors.shape,
            'cpu_usage': self.cpu_usage.shape,
            'ram_usage': self.ram_usage.shape,
            'network_throughput': self.network_throughput.shape,
            'disk_io': self.disk_io.shape,
            'job_runtime': self.job_runtime.shape,
            'queue_length': self.queue_length.shape,
            'active_processes': self.active_processes.shape,
            'timestamp': self.timestamp.shape,
            'node_ids': self.node_ids.shape,
            'job_ids': self.job_ids.shape
        }
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'ClusterMetrics':
        """
        Create ClusterMetrics from DataFrame.
        
        Args:
            df: DataFrame with metrics
            
        Returns:
            ClusterMetrics instance
        """
        return cls(
            gpu_utilization=df['gpu_utilization'].values.reshape(-1, 1),
            memory_utilization=df['memory_utilization'].values.reshape(-1, 1),
            gpu_temperature=df['gpu_temperature'].values.reshape(-1, 1),
            power_consumption=df['power_consumption'].values.reshape(-1, 1),
            ecc_errors=df['ecc_errors'].values.reshape(-1, 1),
            xid_errors=df['xid_errors'].values.reshape(-1, 1),
            cpu_usage=df['cpu_usage'].values.reshape(-1, 1),
            ram_usage=df['ram_usage'].values.reshape(-1, 1),
            network_throughput=df['network_throughput'].values.reshape(-1, 1),
            disk_io=df['disk_io'].values.reshape(-1, 1),
            job_runtime=df['job_runtime'].values.reshape(-1, 1),
            queue_length=df['queue_length'].values.reshape(-1, 1),
            active_processes=df['active_processes'].values.reshape(-1, 1),
            timestamp=df['timestamp'].values.reshape(-1, 1),
            node_ids=df['node_id'].values.reshape(-1, 1),
            job_ids=df['job_id'].values.reshape(-1, 1)
        )


class DataIngestion:
    """
    Data ingestion pipeline for cluster metrics.
    Supports synthetic generation and real trace data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data ingestion.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics = config['features']['metrics']
        self.window_size = config['data']['processing']['window_size']
        self.synthetic_config = config['data']['synthetic']
        self.real_config = config['data']['real']
        self.prometheus_config = config['data']['prometheus']
        
        # Cache for loaded data
        self._cache = {}
        
        logger.info(f"DataIngestion initialized with {len(self.metrics)} metrics")
    
    def load_data(self, source: str = 'synthetic', **kwargs) -> pd.DataFrame:
        """
        Load data from specified source.
        
        Args:
            source: 'synthetic', 'real', or 'prometheus'
            **kwargs: Additional arguments for specific sources
            
        Returns:
            DataFrame with cluster metrics
        """
        cache_key = f"{source}_{hash(str(kwargs))}"
        
        if cache_key in self._cache:
            logger.info(f"Returning cached data for source: {source}")
            return self._cache[cache_key]
        
        if source == 'synthetic':
            df = self._load_synthetic_data(**kwargs)
        elif source == 'real':
            df = self._load_real_data(**kwargs)
        elif source == 'prometheus':
            df = self._load_prometheus_data(**kwargs)
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        # Validate data
        if not self.validate_data(df):
            logger.warning("Data validation failed, but continuing...")
        
        # Cache the data
        self._cache[cache_key] = df
        
        logger.info(f"Loaded {len(df)} data points from {source}")
        return df
    
    def _load_synthetic_data(self, **kwargs) -> pd.DataFrame:
        """
        Load synthetic data.
        
        Args:
            **kwargs: Arguments for synthetic data generation
            
        Returns:
            DataFrame with synthetic metrics
        """
        from .synthetic_generator import SyntheticDataGenerator
        
        # Override config with kwargs
        if kwargs:
            synthetic_config = {**self.synthetic_config, **kwargs}
        else:
            synthetic_config = self.synthetic_config
        
        generator = SyntheticDataGenerator(self.config)
        
        # Override generator settings if provided
        if 'num_nodes' in synthetic_config:
            generator.num_nodes = synthetic_config['num_nodes']
        if 'num_jobs' in synthetic_config:
            generator.num_jobs = synthetic_config['num_jobs']
        if 'time_steps' in synthetic_config:
            generator.time_steps = synthetic_config['time_steps']
        if 'seed' in synthetic_config:
            np.random.seed(synthetic_config['seed'])
        
        cluster_metrics = generator.generate()
        return cluster_metrics.to_dataframe()
    
    def _load_real_data(self, **kwargs) -> pd.DataFrame:
        """
        Load real cluster trace data.
        
        Args:
            **kwargs: Arguments for loading real data
            
        Returns:
            DataFrame with real metrics
        """
        data_path = Path(self.real_config['path'])
        file_pattern = kwargs.get('file_pattern', self.real_config['file_pattern'])
        
        if not data_path.exists():
            logger.warning(f"Real data path does not exist: {data_path}")
            logger.info("Falling back to synthetic data")
            return self._load_synthetic_data(**kwargs)
        
        # Find matching files
        files = list(data_path.glob(file_pattern))
        
        if not files:
            logger.warning(f"No files found matching pattern: {file_pattern}")
            logger.info("Falling back to synthetic data")
            return self._load_synthetic_data(**kwargs)
        
        # Load and concatenate data
        dfs = []
        for file in files:
            try:
                if file.suffix == '.csv':
                    df = pd.read_csv(file)
                elif file.suffix in ['.parquet', '.pqt']:
                    df = pd.read_parquet(file)
                elif file.suffix == '.json':
                    df = pd.read_json(file)
                else:
                    logger.warning(f"Unsupported file format: {file.suffix}")
                    continue
                
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                continue
        
        if not dfs:
            logger.warning("No real data could be loaded")
            return self._load_synthetic_data(**kwargs)
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Take a subset if batch_size is specified
        batch_size = kwargs.get('batch_size', self.real_config.get('batch_size', 1000))
        if len(df) > batch_size:
            df = df.sample(batch_size, random_state=42)
        
        return df
    
    def _load_prometheus_data(self, **kwargs) -> pd.DataFrame:
        """
        Load data from Prometheus.
        
        Args:
            **kwargs: Arguments for Prometheus query
            
        Returns:
            DataFrame with Prometheus metrics
        """
        try:
            import requests
            
            prometheus_url = self.prometheus_config['url']
            query_interval = self.prometheus_config['query_interval']
            
            # Build query
            query = kwargs.get('query', 'gpu_utilization')
            time_range = kwargs.get('time_range', '1h')
            
            # Query Prometheus
            response = requests.get(
                f"{prometheus_url}/api/v1/query",
                params={
                    'query': query,
                    'time': time_range
                },
                timeout=self.prometheus_config.get('timeout', 30)
            )
            
            if response.status_code != 200:
                logger.error(f"Prometheus query failed: {response.status_code}")
                return self._load_synthetic_data(**kwargs)
            
            data = response.json()
            
            # Parse Prometheus response
            if 'data' in data and 'result' in data['data']:
                results = data['data']['result']
                if results:
                    # Convert to DataFrame
                    df_data = []
                    for result in results:
                        metric = result['metric']
                        values = result['values']
                        for timestamp, value in values:
                            df_data.append({
                                'timestamp': float(timestamp),
                                **metric,
                                'value': float(value)
                            })
                    
                    df = pd.DataFrame(df_data)
                    return df
            
            logger.warning("No data from Prometheus, using synthetic")
            return self._load_synthetic_data(**kwargs)
            
        except ImportError:
            logger.warning("Requests library not available, using synthetic")
            return self._load_synthetic_data(**kwargs)
        except Exception as e:
            logger.error(f"Error loading from Prometheus: {e}")
            return self._load_synthetic_data(**kwargs)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required metrics are present.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = self.metrics + ['timestamp', 'node_id', 'job_id']
        missing = set(required_cols) - set(df.columns)
        
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values detected: {null_counts[null_counts > 0].to_dict()}")
        
        # Check data types
        numeric_cols = self.metrics + ['timestamp']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Column {col} is not numeric")
        
        return True
    
    def get_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'metrics_present': [col for col in self.metrics if col in df.columns],
            'metrics_missing': [col for col in self.metrics if col not in df.columns],
            'unique_nodes': len(df['node_id'].unique()) if 'node_id' in df else 0,
            'unique_jobs': len(df['job_id'].unique()) if 'job_id' in df else 0,
            'time_range': {
                'min': df['timestamp'].min() if 'timestamp' in df else None,
                'max': df['timestamp'].max() if 'timestamp' in df else None
            },
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Add column statistics
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        stats['numeric_columns'] = len(numeric_cols)
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def stream_data(self, source: str = 'synthetic', batch_size: int = 100, **kwargs):
        """
        Stream data in batches.
        
        Args:
            source: Data source
            batch_size: Size of each batch
            **kwargs: Additional arguments
            
        Yields:
            DataFrame batches
        """
        df = self.load_data(source, **kwargs)
        
        for i in range(0, len(df), batch_size):
            yield df.iloc[i:i+batch_size]
    
    def save_data(self, df: pd.DataFrame, path: str, format: str = 'parquet') -> None:
        """
        Save data to disk.
        
        Args:
            df: DataFrame to save
            path: Path to save to
            format: Format to save as (parquet, csv, json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            df.to_parquet(path, index=False)
        elif format == 'csv':
            df.to_csv(path, index=False)
        elif format == 'json':
            df.to_json(path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved data to {path} ({format})")


# Convenience function for quick data loading
def load_cluster_data(config: Dict[str, Any], source: str = 'synthetic') -> pd.DataFrame:
    """
    Quick function to load cluster data.
    
    Args:
        config: Configuration dictionary
        source: Data source
        
    Returns:
        DataFrame with cluster metrics
    """
    ingestion = DataIngestion(config)
    return ingestion.load_data(source)