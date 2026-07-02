"""
Configuration management for Cluster Heartbeat.
Handles loading, validation, and environment variable overrides.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager for Cluster Heartbeat.
    Handles loading config from files, environment variables, and defaults.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.merge_order = ['defaults', 'config_file', 'environment']
        
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from all sources.
        
        Returns:
            Merged configuration dictionary
        """
        # Start with defaults
        self.config = self._get_default_config()
        
        # Load from file if exists
        if self.config_path:
            file_config = self._load_from_file(self.config_path)
            if file_config:
                self.config = self._deep_merge(self.config, file_config)
        
        # Override with environment variables
        env_config = self._load_from_environment()
        self.config = self._deep_merge(self.config, env_config)
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'project': {
                'name': 'Cluster Heartbeat',
                'version': '1.0.0',
                'description': 'AI-powered GPU cluster intelligence system',
                'environment': 'development',
                'debug': False
            },
            'data': {
                'synthetic': {
                    'enabled': True,
                    'num_nodes': 10,
                    'num_jobs': 100,
                    'time_steps': 1000,
                    'seed': 42
                },
                'real': {
                    'enabled': False,
                    'dataset': 'alibaba_cluster_trace',
                    'path': '/data/cluster_trace/',
                    'file_pattern': '*.csv',
                    'batch_size': 1000
                },
                'prometheus': {
                    'enabled': False,
                    'url': 'http://prometheus:9090',
                    'query_interval': 60
                },
                'processing': {
                    'window_size': 300,
                    'stride': 60,
                    'sequence_length': 100,
                    'batch_size': 64,
                    'sampling_rate': 1.0,
                    'random_seed': 42,
                    'validation_split': 0.2,
                    'test_split': 0.1
                },
                'storage': {
                    'raw_data_path': 'data/raw/',
                    'processed_data_path': 'data/processed/',
                    'cache_path': 'data/cache/',
                    'feature_path': 'data/features/'
                }
            },
            'features': {
                'metrics': [
                    'gpu_utilization', 'memory_utilization', 'gpu_temperature',
                    'power_consumption', 'ecc_errors', 'xid_errors',
                    'cpu_usage', 'ram_usage', 'network_throughput',
                    'disk_io', 'job_runtime', 'queue_length', 'active_processes'
                ],
                'extraction': {
                    'statistical': ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'skew', 'kurtosis'],
                    'trend': ['slope', 'change_rate', 'acceleration'],
                    'spectral': {'fft_components': 5, 'dominant_frequency': True},
                    'interaction': ['correlations', 'cross_metrics']
                },
                'normalization': {
                    'method': 'standard',
                    'scaling': True,
                    'centering': True
                },
                'reduction': {
                    'enabled': True,
                    'method': 'pca',
                    'components': 0.95,
                    'output_dim': 32
                }
            },
            'model': {
                'fingerprint': {
                    'type': 'autoencoder',
                    'input_dim': None,
                    'latent_dim': 32,
                    'hidden_dims': [64, 128, 64],
                    'activation': 'relu',
                    'dropout_rate': 0.2,
                    'batch_norm': True,
                    'learning_rate': 0.001,
                    'optimizer': 'adam',
                    'weight_decay': 0.00001,
                    'batch_size': 64,
                    'epochs': 100,
                    'early_stopping_patience': 10,
                    'early_stopping_min_delta': 0.0001,
                    'validation_frequency': 1,
                    'loss': {
                        'reconstruction': 'mse',
                        'kl_weight': 0.001
                    },
                    'checkpoint_dir': 'models_checkpoints/',
                    'save_best_only': True,
                    'save_frequency': 10
                },
                'anomaly': {
                    'methods': ['isolation_forest', 'local_outlier_factor', 'reconstruction_error'],
                    'threshold_percentile': 95,
                    'contamination': 0.1,
                    'n_estimators': 100,
                    'max_samples': 0.8,
                    'n_neighbors': 20,
                    'score_ensemble': True,
                    'score_weights': {
                        'isolation_forest': 0.4,
                        'local_outlier_factor': 0.3,
                        'reconstruction_error': 0.3
                    }
                },
                'prediction': {
                    'horizon': 3600,
                    'sequence_length': 300,
                    'failure': {
                        'enabled': True,
                        'model_type': 'lstm',
                        'time_to_failure': True,
                        'confidence_interval': 0.95
                    },
                    'forecasting': {
                        'enabled': True,
                        'horizon_multiple': 24,
                        'confidence_interval': 0.95
                    }
                },
                'scheduler': {
                    'enabled': True,
                    'resource_weights': {'gpu': 1.0, 'memory': 0.8, 'cpu': 0.6, 'network': 0.3},
                    'bin_packing_enabled': True,
                    'bin_packing_threshold': 0.8,
                    'policies': ['resource_fit', 'workload_compatibility', 'performance_isolation', 'failure_awareness'],
                    'kubernetes': {
                        'enabled': False,
                        'namespace': 'default',
                        'scheduler_name': 'cluster-heartbeat'
                    }
                },
                'cost': {
                    'enabled': True,
                    'gpu_hourly_rate': 4.0,
                    'idle_threshold': 0.1,
                    'detection_window': 600,
                    'memory_idle_threshold': 0.5,
                    'alerts': {
                        'idle_gpu_threshold': 2,
                        'cost_waste_threshold': 100.0
                    }
                }
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'debug': False,
                'reload': False,
                'security': {
                    'enabled': False,
                    'secret_key': 'your-secret-key-here',
                    'algorithm': 'HS256',
                    'token_expire_minutes': 1440
                },
                'cors': {
                    'allowed_origins': ['*'],
                    'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                    'allowed_headers': ['Content-Type', 'Authorization', 'X-Request-ID'],
                    'max_age': 3600
                },
                'rate_limit': {
                    'enabled': True,
                    'requests_per_minute': 60,
                    'burst_multiplier': 2
                },
                'docs': {
                    'enabled': True,
                    'title': 'Cluster Heartbeat API',
                    'description': 'AI-powered GPU cluster intelligence system',
                    'version': '1.0.0',
                    'openapi_url': '/openapi.json',
                    'docs_url': '/docs',
                    'redoc_url': '/redoc'
                }
            },
            'logging': {
                'level': 'INFO',
                'format': 'json',
                'handlers': {
                    'console': {'enabled': True, 'level': 'INFO', 'format': 'text'},
                    'file': {'enabled': True, 'level': 'DEBUG', 'path': 'logs/app.log', 
                             'max_size': 10485760, 'backup_count': 10, 'format': 'json'},
                    'metrics': {'enabled': True, 'level': 'INFO', 'path': 'logs/metrics.log'},
                    'audit': {'enabled': True, 'level': 'INFO', 'path': 'logs/audit.log'}
                },
                'rotation': {'when': 'midnight', 'interval': 1, 'backup_count': 30}
            },
            'monitoring': {
                'metrics': {
                    'enabled': True,
                    'prometheus_port': 8001,
                    'metrics_prefix': 'cluster_heartbeat',
                    'custom_metrics': ['cluster_health_score', 'gpu_health_score', 
                                     'failure_risk_score', 'idle_gpu_count', 'cost_savings']
                },
                'health': {
                    'enabled': True,
                    'endpoint': '/api/v1/health',
                    'liveness_endpoint': '/api/v1/health/liveness',
                    'readiness_endpoint': '/api/v1/health/readiness',
                    'checks': ['data_ingestion', 'models', 'api', 'database', 'prometheus']
                },
                'tracing': {
                    'enabled': False,
                    'provider': 'jaeger',
                    'service_name': 'cluster-heartbeat',
                    'sampling_rate': 0.1
                }
            },
            'integrations': {
                'prometheus': {'enabled': True, 'url': 'http://prometheus:9090', 'timeout': 30, 'verify_ssl': False},
                'dcgm': {'enabled': True, 'host': 'dcgm-exporter', 'port': 9400, 
                        'metrics_path': '/metrics', 'collection_interval': 5},
                'kubernetes': {'enabled': False, 'api_server': 'https://kubernetes.default.svc',
                             'token': '', 'verify_ssl': True, 'namespace': 'default', 'config_path': '~/.kube/config'},
                'slack': {'enabled': False, 'webhook_url': '', 'channel': '#cluster-heartbeat-alerts', 'username': 'Cluster Heartbeat'},
                'email': {'enabled': False, 'smtp_host': 'smtp.gmail.com', 'smtp_port': 587,
                         'username': '', 'password': '', 'from': 'alert@cluster-heartbeat.com', 'to': ['admin@example.com']}
            },
            'alerting': {
                'enabled': True,
                'conditions': [
                    {'name': 'high_gpu_temperature', 'metric': 'gpu_temperature', 
                     'threshold': 80, 'operator': '>', 'duration': '5m', 'severity': 'critical'},
                    {'name': 'low_cluster_health', 'metric': 'cluster_health_score',
                     'threshold': 70, 'operator': '<', 'duration': '5m', 'severity': 'warning'},
                    {'name': 'idle_gpu_detected', 'metric': 'idle_gpu_count',
                     'threshold': 2, 'operator': '>', 'duration': '10m', 'severity': 'warning'},
                    {'name': 'high_failure_risk', 'metric': 'failure_risk_score',
                     'threshold': 0.8, 'operator': '>', 'duration': '2m', 'severity': 'critical'}
                ],
                'policies': {
                    'notification_channels': ['slack', 'email'],
                    'grouping': {'enabled': True, 'timeout': '30s'},
                    'silencing': {'enabled': True, 'duration': '1h'}
                }
            },
            'features': {
                'predictive_ops': True,
                'smart_scheduling': True,
                'cost_optimization': True,
                'auto_scaling': False,
                'multi_cluster': False,
                'federated_learning': False,
                'experimental': {
                    'transformer_based': False,
                    'reinforcement_learning': False,
                    'graph_neural_network': False
                }
            }
        }
    
    def _load_from_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix == '.json':
                    return json.load(f) or {}
                else:
                    logger.warning(f"Unsupported config format: {config_path.suffix}")
                    return {}
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary
        """
        env_config = {}
        
        # Map environment variables to config keys
        env_mappings = {
            'PROJECT_ENVIRONMENT': 'project.environment',
            'PROJECT_DEBUG': 'project.debug',
            'API_HOST': 'api.host',
            'API_PORT': 'api.port',
            'API_WORKERS': 'api.workers',
            'API_DEBUG': 'api.debug',
            'API_RELOAD': 'api.reload',
            'LOG_LEVEL': 'logging.level',
            'LOG_FORMAT': 'logging.format',
            'DATA_WINDOW_SIZE': 'data.processing.window_size',
            'DATA_STRIDE': 'data.processing.stride',
            'MODEL_EPOCHS': 'model.fingerprint.epochs',
            'MODEL_BATCH_SIZE': 'model.fingerprint.batch_size',
            'MODEL_LATENT_DIM': 'model.fingerprint.latent_dim',
            'MODEL_LEARNING_RATE': 'model.fingerprint.learning_rate',
            'GPU_HOURLY_RATE': 'model.cost.gpu_hourly_rate',
            'PROMETHEUS_URL': 'integrations.prometheus.url',
            'SECRET_KEY': 'api.security.secret_key',
            'JWT_SECRET': 'api.security.secret_key',
            'SLACK_WEBHOOK': 'integrations.slack.webhook_url',
            'DB_PASSWORD': 'database.postgres.password',
            'REDIS_PASSWORD': 'database.redis.password'
        }
        
        for env_key, config_key in env_mappings.items():
            value = os.environ.get(env_key)
            if value is not None:
                self._set_nested_value(env_config, config_key, self._parse_env_value(value))
        
        return env_config
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Integer
        if value.isdigit():
            return int(value)
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # List (comma separated)
        if ',' in value and not value.startswith('['):
            return [item.strip() for item in value.split(',')]
        
        # JSON
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except:
                pass
        
        # String
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """
        Set nested dictionary value using dot notation.
        
        Args:
            config: Configuration dictionary
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split('.')
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        # Check required fields
        required_fields = [
            'project.name',
            'project.version',
            'api.host',
            'api.port'
        ]
        
        for field in required_fields:
            value = self._get_nested_value(self.config, field)
            if value is None:
                logger.warning(f"Missing required config field: {field}")
        
        # Check numeric ranges
        if self.config.get('api', {}).get('port', 0) not in range(1, 65536):
            logger.warning("API port should be between 1 and 65535")
        
        # Check data paths
        storage = self.config.get('data', {}).get('storage', {})
        for path_key in ['raw_data_path', 'processed_data_path', 'cache_path', 'feature_path']:
            path = storage.get(path_key)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-separated key path
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if not self.config:
            self.load()
        
        value = self._get_nested_value(self.config, key)
        return value if value is not None else default


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    global _config_manager
    
    if config_path is None:
        config_path = os.environ.get('CONFIG_PATH', 'config/config.yaml')
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager.load()


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a specific configuration value.
    
    Args:
        key: Dot-separated key path
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    if _config_manager is None:
        load_config()
    
    return _config_manager.get(key, default)


def reload_config() -> Dict[str, Any]:
    """
    Reload configuration.
    
    Returns:
        Reloaded configuration dictionary
    """
    global _config_manager
    
    if _config_manager:
        return _config_manager.load()
    else:
        return load_config()


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Path to save configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")


# Convenience function for getting config in code
def get_config():
    """Get the loaded configuration."""
    return load_config()


# Export commonly used functions
__all__ = [
    'load_config',
    'get_config_value',
    'reload_config',
    'save_config',
    'get_config',
    'ConfigManager'
]