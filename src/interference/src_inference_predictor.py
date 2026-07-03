"""
Prediction service for Cluster Heartbeat.
Handles model inference, predictions, and result formatting.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
import logging

from ..models.fingerprint import FingerprintAutoencoder, FingerprintTrainer
from ..models.anomaly import AnomalyDetector
from ..features.extractor import FeatureExtractor, WindowFeatures
from ..features.normalizer import FeatureNormalizer
from ..data.preprocessing import DataPreprocessor
from ..config import load_config, get_config_value
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_dir: str = "models_checkpoints"
    config_path: str = "config/config.yaml"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    use_cache: bool = True
    cache_ttl: int = 300  # seconds


@dataclass
class PredictionResult:
    """Container for prediction results."""
    timestamp: datetime = field(default_factory=datetime.now)
    predictions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'predictions': self.predictions,
            'metadata': self.metadata,
            'confidence': self.confidence,
            'processing_time': self.processing_time
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ModelPredictor:
    """
    Base predictor class for loading and using trained models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        
        # Components
        self.fingerprint_model = None
        self.anomaly_detector = None
        self.normalizer = None
        self.feature_extractor = None
        self.preprocessor = None
        
        logger.info(f"ModelPredictor initialized on device: {self.device}")
    
    def load_models(self, model_dir: str = "models_checkpoints") -> 'ModelPredictor':
        """
        Load all trained models from directory.
        
        Args:
            model_dir: Directory containing trained models
            
        Returns:
            Self
        """
        model_dir = Path(model_dir)
        
        try:
            # Load normalizer
            normalizer_path = model_dir / 'normalizer.pkl'
            if normalizer_path.exists():
                self.normalizer = FeatureNormalizer(self.config)
                self.normalizer.load(str(normalizer_path))
                logger.info("Loaded normalizer")
            else:
                logger.warning(f"Normalizer not found at {normalizer_path}")
            
            # Load fingerprint model
            fingerprint_path = model_dir / 'fingerprint_model.pt'
            if fingerprint_path.exists():
                checkpoint = torch.load(fingerprint_path, map_location=self.device)
                input_dim = checkpoint.get('input_dim', 128)
                latent_dim = checkpoint.get('latent_dim', 32)
                hidden_dims = checkpoint.get('hidden_dims', [64, 128, 64])
                
                self.fingerprint_model = FingerprintAutoencoder(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    hidden_dims=hidden_dims
                ).to(self.device)
                self.fingerprint_model.load_state_dict(checkpoint['model_state_dict'])
                self.fingerprint_model.eval()
                logger.info("Loaded fingerprint model")
            else:
                logger.warning(f"Fingerprint model not found at {fingerprint_path}")
            
            # Load anomaly detector
            anomaly_path = model_dir / 'anomaly_detector.pkl'
            if anomaly_path.exists():
                self.anomaly_detector = AnomalyDetector(self.config)
                self.anomaly_detector.load(str(anomaly_path))
                logger.info("Loaded anomaly detector")
            else:
                logger.warning(f"Anomaly detector not found at {anomaly_path}")
            
            # Initialize feature extractor
            self.feature_extractor = FeatureExtractor(self.config)
            self.preprocessor = DataPreprocessor(self.config)
            
            self.is_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            self.is_loaded = False
            raise
        
        return self


class FingerprintPredictor(ModelPredictor):
    """
    Predictor for generating workload fingerprints.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fingerprint predictor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._cache = {}
    
    def generate_fingerprint(self, features: np.ndarray) -> np.ndarray:
        """
        Generate fingerprint from features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Fingerprint vector
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        if self.fingerprint_model is None:
            logger.warning("Fingerprint model not available, using fallback")
            return self._fallback_fingerprint(features)
        
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            _, fingerprint = self.fingerprint_model(x)
            return fingerprint.cpu().numpy()
    
    def _fallback_fingerprint(self, features: np.ndarray) -> np.ndarray:
        """Fallback fingerprint generation using PCA."""
        if len(features) == 0:
            return np.array([])
        
        from sklearn.decomposition import PCA
        n_components = min(32, features.shape[1])
        pca = PCA(n_components=n_components)
        fingerprint = pca.fit_transform(features)
        
        # Normalize
        fingerprint = (fingerprint - np.mean(fingerprint, axis=0)) / (np.std(fingerprint, axis=0) + 1e-8)
        
        return fingerprint
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate fingerprints for a batch of data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Fingerprint matrix
        """
        # Preprocess
        df_clean = self.preprocessor.clean_data(df)
        
        # Extract features
        feature_matrix, _ = self.feature_extractor.extract_all_features(df_clean)
        
        if len(feature_matrix) == 0:
            return np.array([])
        
        # Normalize
        if self.normalizer and self.normalizer.is_fitted:
            features_norm = self.normalizer.transform(feature_matrix)
        else:
            features_norm = self._normalize_fallback(feature_matrix)
        
        # Generate fingerprints
        return self.generate_fingerprint(features_norm)
    
    def _normalize_fallback(self, features: np.ndarray) -> np.ndarray:
        """Fallback normalization."""
        if self.normalizer is not None and self.normalizer.is_fitted:
            return self.normalizer.transform(features)
        
        # Simple normalization
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        return (features - mean) / std
    
    def predict_single(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Generate fingerprint for a single data point.
        
        Args:
            data: Single data point
            
        Returns:
            Fingerprint vector
        """
        df = pd.DataFrame([data])
        fingerprints = self.predict_batch(df)
        return fingerprints[0] if len(fingerprints) > 0 else np.array([])
    
    def get_fingerprint_stats(self, fingerprints: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics of fingerprints.
        
        Args:
            fingerprints: Fingerprint matrix
            
        Returns:
            Statistics dictionary
        """
        if len(fingerprints) == 0:
            return {}
        
        return {
            'shape': fingerprints.shape,
            'mean': float(np.mean(fingerprints)),
            'std': float(np.std(fingerprints)),
            'min': float(np.min(fingerprints)),
            'max': float(np.max(fingerprints)),
            'l2_norm': float(np.mean(np.linalg.norm(fingerprints, axis=1)))
        }


class AnomalyPredictor(FingerprintPredictor):
    """
    Predictor for anomaly detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize anomaly predictor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._reconstruction_cache = {}
    
    def predict_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict anomalies for a batch of data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Anomaly detection results
        """
        start_time = time.time()
        
        # Generate fingerprints
        fingerprints = self.predict_batch(df)
        
        if len(fingerprints) == 0:
            return {
                'error': 'No features extracted',
                'processing_time': time.time() - start_time
            }
        
        # Compute reconstruction errors
        recon_errors = self._compute_reconstruction_errors(df, fingerprints)
        
        # Detect anomalies
        if self.anomaly_detector is not None and self.anomaly_detector.is_fitted:
            results = self.anomaly_detector.predict(fingerprints, recon_errors)
        else:
            results = self._fallback_anomaly_detection(fingerprints, recon_errors)
        
        results['processing_time'] = time.time() - start_time
        
        return results
    
    def _compute_reconstruction_errors(self, df: pd.DataFrame, fingerprints: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction errors.
        
        Args:
            df: Input DataFrame
            fingerprints: Generated fingerprints
            
        Returns:
            Reconstruction errors
        """
        # Preprocess and extract features
        df_clean = self.preprocessor.clean_data(df)
        feature_matrix, _ = self.feature_extractor.extract_all_features(df_clean)
        
        if len(feature_matrix) == 0:
            return np.array([])
        
        # Normalize
        if self.normalizer and self.normalizer.is_fitted:
            features_norm = self.normalizer.transform(feature_matrix)
        else:
            features_norm = self._normalize_fallback(feature_matrix)
        
        # Reconstruct
        if self.fingerprint_model is not None:
            with torch.no_grad():
                x = torch.FloatTensor(features_norm).to(self.device)
                reconstructed, _ = self.fingerprint_model(x)
                reconstructed = reconstructed.cpu().numpy()
            
            # Compute reconstruction errors
            errors = np.linalg.norm(features_norm - reconstructed, axis=1)
            return errors
        
        return np.zeros(len(features_norm))
    
    def _fallback_anomaly_detection(self, fingerprints: np.ndarray, recon_errors: np.ndarray) -> Dict[str, Any]:
        """Fallback anomaly detection using distance-based method."""
        if len(fingerprints) == 0:
            return {
                'scores': [],
                'predictions': [],
                'probabilities': [],
                'threshold': 0
            }
        
        from scipy.spatial.distance import cdist
        
        # Use reconstruction errors as anomaly scores
        if len(recon_errors) == len(fingerprints):
            scores = recon_errors
        else:
            # Use distance from mean
            mean_fp = np.mean(fingerprints, axis=0)
            scores = cdist(fingerprints, [mean_fp]).flatten()
        
        threshold = np.percentile(scores, 95)
        
        return {
            'scores': scores.tolist(),
            'predictions': (scores > threshold).astype(int).tolist(),
            'probabilities': (1 - scores / (scores.max() + 1e-8)).tolist(),
            'threshold': float(threshold)
        }
    
    def predict_failure_risk(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict failure risk for GPUs and nodes.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Failure risk predictions
        """
        anomaly_results = self.predict_anomalies(df)
        
        if 'error' in anomaly_results:
            return anomaly_results
        
        scores = anomaly_results.get('scores', [])
        probabilities = anomaly_results.get('probabilities', [])
        preds = anomaly_results.get('predictions', [])
        
        predictions = []
        for i, score in enumerate(scores):
            risk_score = min(score / 3.0, 1.0)  # Normalize to [0, 1]
            risk_level = "high" if risk_score > 0.7 else "medium" if risk_score > 0.3 else "low"
            
            predictions.append({
                'index': i,
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'is_anomaly': bool(preds[i]) if i < len(preds) else False,
                'confidence': float(probabilities[i]) if i < len(probabilities) else 0.5
            })
        
        return {
            'predictions': predictions,
            'total': len(predictions),
            'anomalies': sum(1 for p in predictions if p['is_anomaly']),
            'avg_risk': float(np.mean([p['risk_score'] for p in predictions])),
            'threshold': anomaly_results.get('threshold', 0)
        }


class HealthPredictor(AnomalyPredictor):
    """
    Predictor for cluster health scores.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize health predictor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
    
    def predict_health(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict health scores for nodes and GPUs.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Health predictions
        """
        start_time = time.time()
        
        # Get anomaly predictions
        anomaly_results = self.predict_anomalies(df)
        
        if 'error' in anomaly_results:
            return anomaly_results
        
        # Compute health scores
        scores = anomaly_results.get('scores', [])
        
        if len(scores) > 0:
            max_score = max(scores) if scores else 1
            health_scores = [100 * (1 - score / (max_score + 1e-8)) for score in scores]
        else:
            health_scores = []
        
        # Node-level health
        node_scores = {}
        if 'node_id' in df.columns:
            for node_id in df['node_id'].unique():
                node_indices = df[df['node_id'] == node_id].index
                node_health = np.mean([health_scores[i] for i in node_indices if i < len(health_scores)])
                if not np.isnan(node_health):
                    node_scores[str(node_id)] = float(node_health)
        
        # Overall health
        avg_health = float(np.mean(health_scores)) if health_scores else 100
        status = self._get_health_status(avg_health)
        
        return {
            'health_scores': health_scores,
            'average_health': avg_health,
            'status': status,
            'node_scores': node_scores,
            'timestamp': datetime.now().isoformat(),
            'processing_time': time.time() - start_time
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
    
    def predict_cluster_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get complete cluster summary.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cluster summary
        """
        health = self.predict_health(df)
        
        return {
            'cluster_status': health.get('status', 'unknown'),
            'average_health': health.get('average_health', 0),
            'total_nodes': len(health.get('node_scores', {})),
            'total_samples': len(df),
            'timestamp': datetime.now().isoformat()
        }


class Predictor:
    """
    Main predictor class that orchestrates all predictions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 model_dir: str = "models_checkpoints"):
        """
        Initialize main predictor.
        
        Args:
            config: Configuration dictionary
            model_dir: Directory containing trained models
        """
        self.config = config or load_config()
        self.model_dir = Path(model_dir)
        
        # Initialize predictors
        self.fingerprint_predictor = FingerprintPredictor(self.config)
        self.anomaly_predictor = AnomalyPredictor(self.config)
        self.health_predictor = HealthPredictor(self.config)
        
        # Load models
        self.load_models()
        
        # Cache
        self._cache = {}
        self._cache_ttl = 300  # seconds
        
        logger.info("Predictor initialized successfully")
    
    def load_models(self) -> 'Predictor':
        """Load all models."""
        self.fingerprint_predictor.load_models(str(self.model_dir))
        self.anomaly_predictor.load_models(str(self.model_dir))
        self.health_predictor.load_models(str(self.model_dir))
        return self
    
    def predict(self, df: pd.DataFrame, prediction_type: str = 'all') -> Dict[str, Any]:
        """
        Make predictions on data.
        
        Args:
            df: Input DataFrame
            prediction_type: Type of prediction ('all', 'fingerprint', 'anomaly', 'health')
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'prediction_type': prediction_type,
            'data_shape': df.shape
        }
        
        if prediction_type in ['all', 'fingerprint']:
            results['fingerprints'] = self.fingerprint_predictor.predict_batch(df).tolist()
        
        if prediction_type in ['all', 'anomaly']:
            results['anomaly'] = self.anomaly_predictor.predict_anomalies(df)
        
        if prediction_type in ['all', 'health']:
            results['health'] = self.health_predictor.predict_health(df)
        
        results['processing_time'] = time.time() - start_time
        
        return results
    
    def predict_from_dict(self, data: Dict[str, Any], prediction_type: str = 'all') -> Dict[str, Any]:
        """
        Make predictions from dictionary data.
        
        Args:
            data: Input data as dictionary
            prediction_type: Type of prediction
            
        Returns:
            Prediction results
        """
        df = pd.DataFrame([data])
        return self.predict(df, prediction_type)
    
    def predict_batch(self, data_list: List[Dict[str, Any]], 
                     prediction_type: str = 'all') -> Dict[str, Any]:
        """
        Make predictions from a list of dictionaries.
        
        Args:
            data_list: List of input data
            prediction_type: Type of prediction
            
        Returns:
            Prediction results
        """
        df = pd.DataFrame(data_list)
        return self.predict(df, prediction_type)
    
    def get_cluster_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get cluster status summary.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cluster status
        """
        return self.health_predictor.predict_cluster_summary(df)
    
    def get_health_score(self, df: pd.DataFrame, node_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health score.
        
        Args:
            df: Input DataFrame
            node_id: Optional node identifier
            
        Returns:
            Health score
        """
        health = self.health_predictor.predict_health(df)
        
        if node_id:
            node_score = health.get('node_scores', {}).get(node_id)
            if node_score is not None:
                return {
                    'node_id': node_id,
                    'health_score': node_score,
                    'status': self.health_predictor._get_health_status(node_score),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'error': f'Node {node_id} not found',
                    'timestamp': datetime.now().isoformat()
                }
        
        return health
    
    def get_failure_risk(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get failure risk predictions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Failure risk predictions
        """
        return self.anomaly_predictor.predict_failure_risk(df)
    
    def get_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get recommendations based on predictions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Recommendations
        """
        # Get predictions
        health = self.health_predictor.predict_health(df)
        anomaly = self.anomaly_predictor.predict_anomalies(df)
        
        recommendations = []
        
        # Health recommendations
        if health.get('status') == 'critical':
            recommendations.append(
                "Cluster health is critical - immediate action required"
            )
        elif health.get('status') == 'degraded':
            recommendations.append(
                "Cluster health is degraded - investigate issues"
            )
        
        # Anomaly recommendations
        if anomaly.get('predictions'):
            anomaly_count = sum(anomaly.get('predictions', []))
            if anomaly_count > 0:
                recommendations.append(
                    f"Detected {anomaly_count} anomalous patterns - investigate"
                )
        
        # Node-specific recommendations
        node_scores = health.get('node_scores', {})
        for node_id, score in node_scores.items():
            if score < 50:
                recommendations.append(
                    f"Node {node_id} has low health ({score:.1f}) - check GPU status"
                )
        
        return {
            'recommendations': recommendations,
            'total': len(recommendations),
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of loaded models.
        
        Returns:
            Model status
        """
        return {
            'fingerprint_model': self.fingerprint_predictor.fingerprint_model is not None,
            'anomaly_detector': self.anomaly_predictor.anomaly_detector is not None,
            'normalizer': self.fingerprint_predictor.normalizer is not None,
            'is_loaded': self.fingerprint_predictor.is_loaded,
            'device': str(self.fingerprint_predictor.device),
            'model_dir': str(self.model_dir)
        }


# Convenience functions
def create_predictor(config_path: Optional[str] = None, 
                     model_dir: str = "models_checkpoints") -> Predictor:
    """
    Create a predictor instance.
    
    Args:
        config_path: Path to configuration file
        model_dir: Directory containing trained models
        
    Returns:
        Predictor instance
    """
    config = load_config(config_path) if config_path else None
    return Predictor(config, model_dir)


def predict_from_data(data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
                     config_path: Optional[str] = None,
                     model_dir: str = "models_checkpoints",
                     prediction_type: str = 'all') -> Dict[str, Any]:
    """
    Quick prediction function.
    
    Args:
        data: Input data (DataFrame, dict, or list of dicts)
        config_path: Path to configuration file
        model_dir: Directory containing trained models
        prediction_type: Type of prediction
        
    Returns:
        Prediction results
    """
    predictor = create_predictor(config_path, model_dir)
    
    if isinstance(data, dict):
        return predictor.predict_from_dict(data, prediction_type)
    elif isinstance(data, list):
        return predictor.predict_batch(data, prediction_type)
    else:
        return predictor.predict(data, prediction_type)