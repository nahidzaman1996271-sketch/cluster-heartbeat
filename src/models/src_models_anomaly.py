"""
Anomaly detection module for Cluster Heartbeat.
Detects deviations from normal behavior patterns.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from scipy.spatial.distance import cdist
import logging
from pathlib import Path
import pickle
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in cluster behavior using multiple methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        anomaly_config = config['model']['anomaly']
        
        # Parameters
        self.threshold_percentile = anomaly_config.get('threshold_percentile', 95)
        self.contamination = anomaly_config.get('contamination', 0.1)
        self.n_estimators = anomaly_config.get('n_estimators', 100)
        self.max_samples = anomaly_config.get('max_samples', 0.8)
        self.n_neighbors = anomaly_config.get('n_neighbors', 20)
        self.score_ensemble = anomaly_config.get('score_ensemble', True)
        self.score_weights = anomaly_config.get('score_weights', {
            'isolation_forest': 0.4,
            'local_outlier_factor': 0.3,
            'reconstruction_error': 0.3
        })
        
        # Models
        self.isolation_forest = None
        self.lof = None
        self.elliptic_envelope = None
        self.threshold = None
        self.is_fitted = False
        
        # State
        self.feature_means = None
        self.feature_stds = None
        
        logger.info("AnomalyDetector initialized")
    
    def fit(self, fingerprints: np.ndarray, 
            reconstruction_errors: Optional[np.ndarray] = None) -> 'AnomalyDetector':
        """
        Fit anomaly detection models on normal data.
        
        Args:
            fingerprints: Normal fingerprints
            reconstruction_errors: Reconstruction errors (optional)
            
        Returns:
            Self
        """
        if len(fingerprints) == 0:
            logger.warning("No data to fit anomaly detector")
            return self
        
        # Store feature statistics
        self.feature_means = np.mean(fingerprints, axis=0)
        self.feature_stds = np.std(fingerprints, axis=0)
        
        # 1. Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(fingerprints)
        
        # 2. Local Outlier Factor
        self.lof = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            novelty=True,
            n_jobs=-1
        )
        self.lof.fit(fingerprints)
        
        # 3. Elliptic Envelope
        try:
            self.elliptic_envelope = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
            self.elliptic_envelope.fit(fingerprints)
        except Exception as e:
            logger.warning(f"Elliptic Envelope failed: {e}")
            self.elliptic_envelope = None
        
        # Compute anomaly scores for threshold
        scores = self._compute_anomaly_scores(fingerprints, reconstruction_errors)
        self.threshold = np.percentile(scores, self.threshold_percentile)
        
        self.is_fitted = True
        logger.info(f"Anomaly detector fitted with threshold={self.threshold:.4f}")
        
        return self
    
    def predict(self, fingerprints: np.ndarray,
                reconstruction_errors: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Predict anomalies for input fingerprints.
        
        Args:
            fingerprints: Input fingerprints
            reconstruction_errors: Reconstruction errors (optional)
            
        Returns:
            Dictionary with anomaly scores and predictions
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector must be fitted before prediction")
        
        if len(fingerprints) == 0:
            return {
                'scores': [],
                'predictions': [],
                'probabilities': [],
                'threshold': self.threshold
            }
        
        # Compute anomaly scores
        scores = self._compute_anomaly_scores(fingerprints, reconstruction_errors)
        
        # Binary predictions
        predictions = (scores > self.threshold).astype(int)
        
        # Compute anomaly probabilities using Gaussian model
        if len(scores) > 1:
            mean = np.mean(scores)
            std = np.std(scores) + 1e-8
            probabilities = 1 - stats.norm.cdf(scores, loc=mean, scale=std)
            # Clip to [0, 1]
            probabilities = np.clip(probabilities, 0, 1)
        else:
            probabilities = np.ones_like(scores) * 0.5
        
        return {
            'scores': scores.tolist(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'threshold': float(self.threshold)
        }
    
    def _compute_anomaly_scores(self, fingerprints: np.ndarray,
                                reconstruction_errors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute anomaly scores using ensemble of methods.
        
        Args:
            fingerprints: Input fingerprints
            reconstruction_errors: Reconstruction errors (optional)
            
        Returns:
            Anomaly scores
        """
        n_samples = len(fingerprints)
        scores = np.zeros(n_samples)
        score_count = 0
        
        # 1. Isolation Forest scores
        if self.isolation_forest is not None:
            if_scores = -self.isolation_forest.score_samples(fingerprints)
            # Normalize
            if_scores_norm = (if_scores - np.mean(if_scores)) / (np.std(if_scores) + 1e-8)
            scores += self.score_weights.get('isolation_forest', 0.4) * if_scores_norm
            score_count += 1
        
        # 2. LOF scores
        if self.lof is not None:
            lof_scores = -self.lof.score_samples(fingerprints)
            # Normalize
            lof_scores_norm = (lof_scores - np.mean(lof_scores)) / (np.std(lof_scores) + 1e-8)
            scores += self.score_weights.get('local_outlier_factor', 0.3) * lof_scores_norm
            score_count += 1
        
        # 3. Elliptic Envelope scores
        if self.elliptic_envelope is not None:
            ee_scores = -self.elliptic_envelope.score_samples(fingerprints)
            ee_scores_norm = (ee_scores - np.mean(ee_scores)) / (np.std(ee_scores) + 1e-8)
            scores += self.score_weights.get('elliptic_envelope', 0.0) * ee_scores_norm
            score_count += 1
        
        # 4. Reconstruction error (if provided)
        if reconstruction_errors is not None:
            # Normalize
            re_scores_norm = (reconstruction_errors - np.mean(reconstruction_errors)) / (np.std(reconstruction_errors) + 1e-8)
            scores += self.score_weights.get('reconstruction_error', 0.3) * re_scores_norm
            score_count += 1
        
        # Average scores
        if score_count > 0:
            scores /= score_count
        
        # Ensure non-negative
        scores = np.maximum(scores, 0)
        
        return scores
    
    def compute_reconstruction_error(self, original: np.ndarray,
                                    reconstructed: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error between original and reconstructed.
        
        Args:
            original: Original features
            reconstructed: Reconstructed features
            
        Returns:
            Reconstruction errors
        """
        errors = np.linalg.norm(original - reconstructed, axis=1)
        return errors
    
    def get_anomaly_summary(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of anomaly detection results.
        
        Args:
            predictions: Prediction dictionary from predict()
            
        Returns:
            Summary dictionary
        """
        scores = predictions.get('scores', [])
        preds = predictions.get('predictions', [])
        probs = predictions.get('probabilities', [])
        
        if not scores:
            return {'total': 0, 'anomalies': 0, 'avg_score': 0}
        
        return {
            'total': len(scores),
            'anomalies': sum(preds),
            'anomaly_ratio': sum(preds) / len(preds) if preds else 0,
            'avg_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'avg_probability': float(np.mean(probs)),
            'threshold': float(predictions.get('threshold', 0))
        }
    
    def save(self, path: str) -> None:
        """
        Save the anomaly detector.
        
        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'isolation_forest': self.isolation_forest,
                'lof': self.lof,
                'elliptic_envelope': self.elliptic_envelope,
                'threshold': self.threshold,
                'feature_means': self.feature_means,
                'feature_stds': self.feature_stds,
                'is_fitted': self.is_fitted,
                'config': self.config
            }, f)
        
        logger.info(f"Anomaly detector saved to {path}")
    
    def load(self, path: str) -> 'AnomalyDetector':
        """
        Load the anomaly detector.
        
        Args:
            path: Path to load from
            
        Returns:
            Self
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.isolation_forest = data['isolation_forest']
        self.lof = data['lof']
        self.elliptic_envelope = data.get('elliptic_envelope', None)
        self.threshold = data['threshold']
        self.feature_means = data.get('feature_means', None)
        self.feature_stds = data.get('feature_stds', None)
        self.is_fitted = data['is_fitted']
        
        logger.info(f"Anomaly detector loaded from {path}")
        return self


# Convenience function
def create_anomaly_detector(config: Dict[str, Any]) -> AnomalyDetector:
    """
    Quick function to create anomaly detector.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AnomalyDetector instance
    """
    return AnomalyDetector(config)