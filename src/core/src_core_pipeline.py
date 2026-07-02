"""
Main pipeline that orchestrates all components of Cluster Heartbeat.
Links data ingestion, feature extraction, model inference, and services.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import torch
import logging
from datetime import datetime
import json
import time

from ..data.ingestion import DataIngestion, ClusterMetrics
from ..data.preprocessing import DataPreprocessor
from ..features.extractor import FeatureExtractor, WindowFeatures
from ..features.normalizer import FeatureNormalizer
from ..models.fingerprint import FingerprintTrainer, FingerprintAutoencoder
from ..models.anomaly import AnomalyDetector
from ..models.scheduler import SmartScheduler, SchedulingRecommendation
from ..models.cost_optimizer import CostOptimizer, CostSavingRecommendation
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ClusterHeartbeatPipeline:
    """
    End-to-end pipeline for Cluster Heartbeat.
    Integrates all components into a single workflow.
    """
    
    def __init__(self, config: Dict[str, Any], model_dir: Optional[str] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration dictionary
            model_dir: Directory containing trained models
        """
        self.config = config
        self.model_dir = Path(model_dir) if model_dir else Path('models_checkpoints')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_ingestion = DataIngestion(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.feature_extractor = FeatureExtractor(config)
        self.normalizer = FeatureNormalizer(config)
        self.fingerprint_trainer = None
        self.fingerprint_model = None
        self.anomaly_detector = None
        self.scheduler = SmartScheduler(config)
        self.cost_optimizer = CostOptimizer(config)
        
        # State
        self.is_initialized = False
        self.fingerprints_cache = {}
        self.metrics_cache = None
        self.last_results = None
        self.pipeline_stats = {
            'total_processed': 0,
            'total_errors': 0,
            'processing_times': [],
            'last_processed_time': None
        }
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Pipeline initialized on device: {self.device}")
    
    def initialize(self, load_models: bool = True) -> 'ClusterHeartbeatPipeline':
        """
        Initialize all components and load models.
        
        Args:
            load_models: Whether to load trained models
            
        Returns:
            Self
        """
        try:
            if load_models:
                self._load_models()
            else:
                self._initialize_models()
            
            self.is_initialized = True
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            raise
        
        return self
    
    def _initialize_models(self):
        """Initialize models without loading (for training)."""
        self.fingerprint_model = None
        self.fingerprint_trainer = FingerprintTrainer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        logger.info("Models initialized (not loaded)")
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            # Load normalizer
            normalizer_path = self.model_dir / 'normalizer.pkl'
            if normalizer_path.exists():
                self.normalizer.load(str(normalizer_path))
                logger.info("Loaded normalizer")
            else:
                logger.warning("Normalizer not found, will use default")
            
            # Load fingerprint model
            fingerprint_path = self.model_dir / 'fingerprint_model.pt'
            if fingerprint_path.exists():
                # Load checkpoint to get config
                checkpoint = torch.load(fingerprint_path, map_location=self.device)
                input_dim = checkpoint.get('input_dim', 128)
                latent_dim = checkpoint.get('latent_dim', 32)
                hidden_dims = checkpoint.get('hidden_dims', [64, 128, 64])
                
                # Rebuild model
                self.fingerprint_model = FingerprintAutoencoder(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    hidden_dims=hidden_dims
                ).to(self.device)
                self.fingerprint_model.load_state_dict(checkpoint['model_state_dict'])
                self.fingerprint_model.eval()
                logger.info("Loaded fingerprint model")
            else:
                logger.warning("Fingerprint model not found, will use fallback")
            
            # Load anomaly detector
            anomaly_path = self.model_dir / 'anomaly_detector.pkl'
            if anomaly_path.exists():
                self.anomaly_detector = AnomalyDetector(self.config)
                self.anomaly_detector.load(str(anomaly_path))
                logger.info("Loaded anomaly detector")
            else:
                logger.warning("Anomaly detector not found, will use default")
            
        except Exception as e:
            logger.warning(f"Could not load all models: {e}")
            logger.info("Using fallback mode")
    
    def process_batch(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of data through the entire pipeline.
        
        Args:
            data: Input DataFrame or dictionary with cluster metrics
            
        Returns:
            Dictionary with all analysis results
        """
        start_time = time.time()
        
        if not self.is_initialized:
            self.initialize(load_models=True)
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # 1. Validate data
            if not self.data_ingestion.validate_data(df):
                logger.warning("Data validation failed, attempting to continue...")
            
            # 2. Preprocess data
            df_clean = self.data_preprocessor.clean_data(df)
            
            # 3. Extract features
            feature_matrix, window_features = self.feature_extractor.extract_all_features(df_clean)
            
            if len(feature_matrix) == 0:
                logger.warning("No features extracted from data")
                return self._empty_results()
            
            # 4. Normalize features
            if self.normalizer.is_fitted:
                features_norm = self.normalizer.transform(feature_matrix)
            else:
                features_norm = self.normalizer.fit_transform(feature_matrix)
            
            # 5. Generate fingerprints
            fingerprints = self._generate_fingerprints(features_norm)
            
            # 6. Detect anomalies
            anomaly_results = self._detect_anomalies(fingerprints)
            
            # 7. Generate scheduling recommendations
            scheduling_results = self._generate_scheduling(fingerprints, df_clean)
            
            # 8. Analyze cost optimization
            cost_results = self._analyze_costs(df_clean)
            
            # 9. Compute health scores
            health_scores = self._compute_health_scores(fingerprints, anomaly_results)
            
            # 10. Combine results
            results = {
                'timestamp': datetime.now().isoformat(),
                'fingerprints': fingerprints.tolist() if len(fingerprints) > 0 else [],
                'anomaly_results': anomaly_results,
                'scheduling': scheduling_results,
                'cost_optimization': cost_results,
                'health_scores': health_scores,
                'summary': self._generate_summary(health_scores, anomaly_results, cost_results),
                'pipeline_stats': {
                    'processing_time': time.time() - start_time,
                    'total_samples': len(df),
                    'feature_dim': features_norm.shape[1] if len(features_norm) > 0 else 0
                }
            }
            
            # Cache results
            self.fingerprints_cache = fingerprints
            self.metrics_cache = df
            self.last_results = results
            self.pipeline_stats['total_processed'] += len(df)
            self.pipeline_stats['processing_times'].append(time.time() - start_time)
            self.pipeline_stats['last_processed_time'] = datetime.now()
            
            logger.info(f"Processed {len(df)} samples in {time.time() - start_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            self.pipeline_stats['total_errors'] += 1
            return self._error_results(str(e))
    
    def _generate_fingerprints(self, features: np.ndarray) -> np.ndarray:
        """
        Generate fingerprints using trained model or fallback.
        
        Args:
            features: Normalized feature matrix
            
        Returns:
            Fingerprints
        """
        if len(features) == 0:
            return np.array([])
        
        if self.fingerprint_model is not None:
            try:
                # Use trained model
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).to(self.device)
                    _, fingerprints = self.fingerprint_model(features_tensor)
                    return fingerprints.cpu().numpy()
            except Exception as e:
                logger.warning(f"Model inference failed: {e}, using fallback")
        
        # Fallback: use PCA-like dimensionality reduction
        logger.info("Using fallback fingerprint generation (PCA-style)")
        from sklearn.decomposition import PCA
        
        n_components = min(self.config['features'].get('embedding_dim', 32), features.shape[1])
        pca = PCA(n_components=n_components)
        fingerprints = pca.fit_transform(features)
        
        # Normalize fingerprints
        fingerprints = (fingerprints - np.mean(fingerprints, axis=0)) / (np.std(fingerprints, axis=0) + 1e-8)
        
        return fingerprints
    
    def _detect_anomalies(self, fingerprints: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies using trained detector.
        
        Args:
            fingerprints: Workload fingerprints
            
        Returns:
            Anomaly detection results
        """
        if len(fingerprints) == 0:
            return {
                'scores': [],
                'predictions': [],
                'probabilities': [],
                'threshold': 0,
                'summary': {'total': 0, 'anomalies': 0}
            }
        
        if self.anomaly_detector is not None and self.anomaly_detector.is_fitted:
            try:
                results = self.anomaly_detector.predict(fingerprints)
                results['summary'] = self.anomaly_detector.get_anomaly_summary(results)
                return results
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}, using fallback")
        
        # Fallback: use distance-based anomaly detection
        logger.info("Using fallback anomaly detection")
        from scipy.spatial.distance import cdist
        
        mean_fp = np.mean(fingerprints, axis=0)
        distances = cdist(fingerprints, [mean_fp]).flatten()
        threshold = np.percentile(distances, 95)
        
        results = {
            'scores': distances.tolist(),
            'predictions': (distances > threshold).astype(int).tolist(),
            'probabilities': (1 - distances / (distances.max() + 1e-8)).tolist(),
            'threshold': float(threshold),
            'summary': {
                'total': len(distances),
                'anomalies': int(np.sum(distances > threshold)),
                'anomaly_ratio': float(np.mean(distances > threshold)),
                'avg_score': float(np.mean(distances))
            }
        }
        return results
    
    def _generate_scheduling(self, fingerprints: np.ndarray, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate scheduling recommendations.
        
        Args:
            fingerprints: Workload fingerprints
            data: Input data
            
        Returns:
            List of scheduling recommendations
        """
        try:
            # Prepare jobs
            jobs = []
            for i, (idx, row) in enumerate(data.iterrows()):
                job = {
                    'id': row.get('job_id', f'job_{i}'),
                    'fingerprint': fingerprints[i] if i < len(fingerprints) else np.zeros(32),
                    'priority': row.get('priority', 0)
                }
                jobs.append(job)
            
            # Get recommendations
            recommendations = self.scheduler.generate_schedule(jobs)
            
            # Convert to dict
            return [self._recommendation_to_dict(rec) for rec in recommendations]
            
        except Exception as e:
            logger.warning(f"Scheduling failed: {e}")
            return []
    
    def _analyze_costs(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cost optimization opportunities.
        
        Args:
            data: Input data
            
        Returns:
            Cost analysis results
        """
        try:
            # Prepare metrics for cost optimizer
            metrics = self._prepare_cost_metrics(data)
            
            # Get recommendations
            recommendations = self.cost_optimizer.analyze_cost_optimization(metrics)
            
            # Generate report
            return self.cost_optimizer.generate_savings_report(recommendations)
            
        except Exception as e:
            logger.warning(f"Cost analysis failed: {e}")
            return {
                'summary': {
                    'total_idle_gpus': 0,
                    'total_cost_wasted': 0,
                    'potential_savings': 0
                },
                'idle_gpus': [],
                'recommendations': ['Cost analysis unavailable']
            }
    
    def _compute_health_scores(self, fingerprints: np.ndarray, anomaly_results: Dict) -> Dict[str, Any]:
        """
        Compute health scores for nodes and GPUs.
        
        Args:
            fingerprints: Workload fingerprints
            anomaly_results: Anomaly detection results
            
        Returns:
            Health scores
        """
        try:
            scores = anomaly_results.get('scores', [])
            probabilities = anomaly_results.get('probabilities', [])
            
            # Compute health scores (inverse of anomaly probability)
            if len(scores) > 0:
                max_score = max(scores) if scores else 1
                health_scores = [100 * (1 - score / (max_score + 1e-8)) for score in scores]
            else:
                health_scores = [100] * len(fingerprints)
            
            # Compute node-level health
            node_scores = {}
            if self.metrics_cache is not None and 'node_id' in self.metrics_cache.columns:
                node_ids = self.metrics_cache['node_id'].values
                unique_nodes = np.unique(node_ids)
                
                for node in unique_nodes:
                    node_indices = np.where(np.array(node_ids) == node)[0]
                    if len(node_indices) > 0:
                        # Get health scores for this node
                        node_health_indices = [i for i in node_indices if i < len(health_scores)]
                        if node_health_indices:
                            node_health = np.mean([health_scores[i] for i in node_health_indices])
                            node_scores[str(node)] = float(node_health)
            
            return {
                'node_scores': node_scores,
                'health_scores': health_scores,
                'average_health': float(np.mean(health_scores)) if health_scores else 100,
                'status': self._get_health_status(np.mean(health_scores)) if health_scores else 'unknown'
            }
            
        except Exception as e:
            logger.warning(f"Health score computation failed: {e}")
            return {
                'node_scores': {},
                'health_scores': [100],
                'average_health': 100,
                'status': 'unknown'
            }
    
    def _get_health_status(self, score: float) -> str:
        """Get health status based on score."""
        if score >= 80:
            return 'healthy'
        elif score >= 60:
            return 'warning'
        elif score >= 40:
            return 'degraded'
        else:
            return 'critical'
    
    def _generate_summary(self, health_scores: Dict, anomaly_results: Dict, cost_results: Dict) -> Dict[str, Any]:
        """Generate overall summary."""
        return {
            'total_nodes': len(health_scores.get('node_scores', {})),
            'average_health': health_scores.get('average_health', 100),
            'health_status': health_scores.get('status', 'unknown'),
            'anomaly_count': sum(anomaly_results.get('predictions', [])),
            'anomaly_ratio': float(np.mean(anomaly_results.get('predictions', [0]))),
            'idle_gpus': cost_results.get('summary', {}).get('total_idle_gpus', 0),
            'cost_savings': cost_results.get('summary', {}).get('potential_savings', 0),
            'total_cost_wasted': cost_results.get('summary', {}).get('total_cost_wasted', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def _recommendation_to_dict(self, rec: SchedulingRecommendation) -> Dict[str, Any]:
        """Convert SchedulingRecommendation to dict."""
        return {
            'job_id': rec.job_id,
            'recommended_node': rec.recommended_node,
            'current_node': rec.current_node,
            'score': rec.score,
            'reason': rec.reason,
            'resource_match': rec.resource_match,
            'alternative_nodes': rec.alternative_nodes,
            'priority': rec.priority,
            'timestamp': rec.timestamp
        }
    
    def _prepare_cost_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare metrics for cost optimizer."""
        gpu_metrics = {}
        
        # Group by node_id
        if 'node_id' in data.columns:
            for node_id, group in data.groupby('node_id'):
                node_id_str = str(node_id)
                gpu_metrics[node_id_str] = {}
                
                for idx, row in group.iterrows():
                    gpu_id = idx % 4  # Assume 4 GPUs per node
                    gpu_metrics[node_id_str][gpu_id] = {
                        'utilization': float(row.get('gpu_utilization', 0.5)),
                        'memory_utilization': float(row.get('memory_utilization', 0.5)),
                        'temperature': float(row.get('gpu_temperature', 65)),
                        'power_consumption': float(row.get('power_consumption', 100))
                    }
        
        return {'gpu_metrics': gpu_metrics}
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results."""
        return {
            'timestamp': datetime.now().isoformat(),
            'fingerprints': [],
            'anomaly_results': {
                'scores': [],
                'predictions': [],
                'probabilities': [],
                'threshold': 0,
                'summary': {'total': 0, 'anomalies': 0}
            },
            'scheduling': [],
            'cost_optimization': {
                'summary': {'total_idle_gpus': 0, 'total_cost_wasted': 0, 'potential_savings': 0},
                'idle_gpus': [],
                'recommendations': ['No data available']
            },
            'health_scores': {
                'node_scores': {},
                'health_scores': [],
                'average_health': 100,
                'status': 'unknown'
            },
            'summary': {
                'total_nodes': 0,
                'average_health': 100,
                'health_status': 'unknown',
                'anomaly_count': 0,
                'anomaly_ratio': 0,
                'idle_gpus': 0,
                'cost_savings': 0,
                'total_cost_wasted': 0
            },
            'pipeline_stats': {
                'processing_time': 0,
                'total_samples': 0,
                'feature_dim': 0
            }
        }
    
    def _error_results(self, error: str) -> Dict[str, Any]:
        """Return error results."""
        results = self._empty_results()
        results['error'] = error
        results['summary']['error'] = error
        return results
    
    def process_single_node(self, node_id: str) -> Dict[str, Any]:
        """
        Process data for a single node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node-specific analysis results
        """
        if self.metrics_cache is None or len(self.metrics_cache) == 0:
            return {'error': 'No cached metrics available'}
        
        # Filter data for node
        if 'node_id' in self.metrics_cache.columns:
            node_data = self.metrics_cache[self.metrics_cache['node_id'] == node_id]
        else:
            return {'error': 'No node_id column in cached data'}
        
        if len(node_data) == 0:
            return {'error': f'Node {node_id} not found'}
        
        # Process filtered data
        results = self.process_batch(node_data)
        results['node_id'] = node_id
        
        return results
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get formatted data for dashboard.
        
        Returns:
            Dashboard-ready JSON
        """
        if self.last_results is None:
            return {'error': 'No data processed yet'}
        
        return {
            'cluster_summary': self._get_cluster_summary(),
            'health_scores': self._get_health_scores(),
            'anomalies': self._get_anomalies(),
            'scheduling_recommendations': self._get_scheduling_recommendations(),
            'cost_savings': self._get_cost_savings(),
            'timeseries_metrics': self._get_timeseries_metrics(),
            'pipeline_stats': self.pipeline_stats
        }
    
    def _get_cluster_summary(self) -> Dict[str, Any]:
        """Get cluster summary for dashboard."""
        if self.last_results is None:
            return {'nodes': 0, 'gpus': 0, 'utilization': 0}
        
        summary = self.last_results.get('summary', {})
        return {
            'nodes': summary.get('total_nodes', 0),
            'gpus': self.pipeline_stats.get('total_processed', 0),
            'average_health': summary.get('average_health', 0),
            'health_status': summary.get('health_status', 'unknown'),
            'anomalies': summary.get('anomaly_count', 0),
            'idle_gpus': summary.get('idle_gpus', 0),
            'potential_savings': summary.get('cost_savings', 0),
            'timestamp': self.last_results.get('timestamp')
        }
    
    def _get_health_scores(self) -> Dict[str, Any]:
        """Get health scores for dashboard."""
        if self.last_results is None:
            return {'status': 'unknown', 'score': 0}
        
        health = self.last_results.get('health_scores', {})
        return {
            'status': health.get('status', 'unknown'),
            'score': health.get('average_health', 0),
            'node_scores': health.get('node_scores', {})
        }
    
    def _get_anomalies(self) -> Dict[str, Any]:
        """Get anomalies for dashboard."""
        if self.last_results is None:
            return {'count': 0, 'details': []}
        
        anomaly = self.last_results.get('anomaly_results', {})
        return {
            'count': int(np.sum(anomaly.get('predictions', []))),
            'ratio': anomaly.get('summary', {}).get('anomaly_ratio', 0),
            'details': []
        }
    
    def _get_scheduling_recommendations(self) -> Dict[str, Any]:
        """Get scheduling recommendations for dashboard."""
        if self.last_results is None:
            return {'recommendations': [], 'total': 0}
        
        scheduling = self.last_results.get('scheduling', [])
        return {
            'recommendations': scheduling,
            'total': len(scheduling)
        }
    
    def _get_cost_savings(self) -> Dict[str, Any]:
        """Get cost savings for dashboard."""
        if self.last_results is None:
            return {'potential_savings': 0, 'idle_gpus': 0}
        
        cost = self.last_results.get('cost_optimization', {})
        summary = cost.get('summary', {})
        return {
            'potential_savings': summary.get('potential_savings', 0),
            'idle_gpus': summary.get('total_idle_gpus', 0),
            'total_wasted': summary.get('total_cost_wasted', 0),
            'recommendations': cost.get('recommendations', [])
        }
    
    def _get_timeseries_metrics(self) -> Dict[str, Any]:
        """Get time series metrics for dashboard."""
        if self.metrics_cache is None:
            return {'timestamps': [], 'metrics': {}}
        
        df = self.metrics_cache
        return {
            'timestamps': df.get('timestamp', []).tolist() if 'timestamp' in df else [],
            'metrics': {
                'gpu_utilization': df.get('gpu_utilization', []).tolist(),
                'memory_utilization': df.get('memory_utilization', []).tolist(),
                'temperature': df.get('gpu_temperature', []).tolist()
            }
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'total_processed': self.pipeline_stats['total_processed'],
            'total_errors': self.pipeline_stats['total_errors'],
            'avg_processing_time': float(np.mean(self.pipeline_stats['processing_times'])),
            'last_processed_time': self.pipeline_stats['last_processed_time'].isoformat() 
                if self.pipeline_stats['last_processed_time'] else None,
            'is_initialized': self.is_initialized,
            'device': str(self.device),
            'models_loaded': {
                'fingerprint': self.fingerprint_model is not None,
                'anomaly_detector': self.anomaly_detector is not None and self.anomaly_detector.is_fitted
            }
        }
    
    def shutdown(self):
        """Clean up resources."""
        self.is_initialized = False
        if self.fingerprint_model is not None:
            del self.fingerprint_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Pipeline shutdown complete")
    
    def save_models(self) -> None:
        """Save all models to disk."""
        if self.fingerprint_trainer and self.fingerprint_trainer.model:
            self.fingerprint_trainer.save_checkpoint('fingerprint_model.pt')
        
        if self.anomaly_detector and self.anomaly_detector.is_fitted:
            self.anomaly_detector.save(str(self.model_dir / 'anomaly_detector.pkl'))
        
        if self.normalizer and self.normalizer.is_fitted:
            self.normalizer.save(str(self.model_dir / 'normalizer.pkl'))
        
        logger.info(f"Models saved to {self.model_dir}")


# Convenience function
def create_pipeline(config: Dict[str, Any]) -> ClusterHeartbeatPipeline:
    """
    Quick function to create pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ClusterHeartbeatPipeline instance
    """
    return ClusterHeartbeatPipeline(config)