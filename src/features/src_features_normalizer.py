"""
Feature normalization module for Cluster Heartbeat.
Handles normalization and standardization of features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
import pickle
import logging
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureNormalizer:
    """
    Normalizes features for fingerprint generation.
    Supports various normalization methods.
    """
    
    def __init__(self, config: Dict[str, Any], method: str = 'standard'):
        """
        Initialize feature normalizer.
        
        Args:
            config: Configuration dictionary
            method: Normalization method
        """
        self.config = config
        self.method = method
        self.scaler = None
        self.feature_means = None
        self.feature_stds = None
        self.is_fitted = False
        self.feature_names = None
        
        # Get method from config if not specified
        if method == 'standard' and 'normalization' in config.get('features', {}):
            self.method = config['features']['normalization'].get('method', 'standard')
        
        # Initialize scaler
        self._initialize_scaler()
        
        logger.info(f"FeatureNormalizer initialized with method: {self.method}")
    
    def _initialize_scaler(self) -> None:
        """Initialize the appropriate scaler."""
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'maxabs':
            self.scaler = MaxAbsScaler()
        elif self.method == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal')
        elif self.method == 'power':
            self.scaler = PowerTransformer(method='yeo-johnson')
        else:
            logger.warning(f"Unknown normalization method: {self.method}, using standard")
            self.scaler = StandardScaler()
    
    def fit(self, features: Union[np.ndarray, pd.DataFrame]) -> 'FeatureNormalizer':
        """
        Fit the normalizer on training data.
        
        Args:
            features: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        # Convert DataFrame to numpy if needed
        if isinstance(features, pd.DataFrame):
            self.feature_names = features.columns.tolist()
            features = features.values
        
        # Fit scaler
        self.scaler.fit(features)
        
        # Store statistics
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        self.is_fitted = True
        
        logger.info(f"Normalizer fitted on {features.shape[1]} features")
        return self
    
    def transform(self, features: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform features using fitted normalizer.
        
        Args:
            features: Feature matrix to transform
            
        Returns:
            Normalized features
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        # Convert DataFrame to numpy if needed
        is_dataframe = isinstance(features, pd.DataFrame)
        if is_dataframe:
            feature_names = features.columns.tolist()
            features = features.values
        
        # Transform
        transformed = self.scaler.transform(features)
        
        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            transformed = pd.DataFrame(
                transformed,
                columns=feature_names,
                index=range(len(transformed))
            )
        
        return transformed
    
    def fit_transform(self, features: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit and transform in one step.
        
        Args:
            features: Feature matrix
            
        Returns:
            Normalized features
        """
        return self.fit(features).transform(features)
    
    def inverse_transform(self, features: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Inverse transform normalized features.
        
        Args:
            features: Normalized features
            
        Returns:
            Original scale features
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")
        
        # Convert DataFrame to numpy if needed
        is_dataframe = isinstance(features, pd.DataFrame)
        if is_dataframe:
            feature_names = features.columns.tolist()
            features = features.values
        
        # Inverse transform
        transformed = self.scaler.inverse_transform(features)
        
        # Convert back to DataFrame if input was DataFrame
        if is_dataframe:
            transformed = pd.DataFrame(
                transformed,
                columns=feature_names,
                index=range(len(transformed))
            )
        
        return transformed
    
    def normalize_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize a batch of features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Normalized features
        """
        if not self.is_fitted:
            return self.fit_transform(features)
        return self.transform(features)
    
    def get_scaling_parameters(self) -> Dict[str, Any]:
        """
        Get scaling parameters.
        
        Returns:
            Dictionary with scaling parameters
        """
        if not self.is_fitted:
            return {}
        
        params = {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'mean': self.feature_means.tolist() if self.feature_means is not None else None,
            'std': self.feature_stds.tolist() if self.feature_stds is not None else None,
            'n_features': len(self.feature_means) if self.feature_means is not None else 0
        }
        
        # Add scaler-specific parameters
        if hasattr(self.scaler, 'mean_'):
            params['scaler_mean'] = self.scaler.mean_.tolist()
        if hasattr(self.scaler, 'scale_'):
            params['scaler_scale'] = self.scaler.scale_.tolist()
        if hasattr(self.scaler, 'min_'):
            params['scaler_min'] = self.scaler.min_.tolist()
        if hasattr(self.scaler, 'data_min_'):
            params['data_min'] = self.scaler.data_min_.tolist()
        if hasattr(self.scaler, 'data_max_'):
            params['data_max'] = self.scaler.data_max_.tolist()
        
        return params
    
    def save(self, path: str) -> None:
        """
        Save the normalizer to disk.
        
        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_means': self.feature_means,
                'feature_stds': self.feature_stds,
                'method': self.method,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names
            }, f)
        
        logger.info(f"Normalizer saved to {path}")
    
    def load(self, path: str) -> 'FeatureNormalizer':
        """
        Load the normalizer from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            Self
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.scaler = data['scaler']
        self.feature_means = data['feature_means']
        self.feature_stds = data['feature_stds']
        self.method = data['method']
        self.is_fitted = data['is_fitted']
        self.feature_names = data.get('feature_names', None)
        
        logger.info(f"Normalizer loaded from {path}")
        return self
    
    def validate_normalized(self, features: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Validate that features are properly normalized.
        
        Args:
            features: Feature matrix
            tolerance: Tolerance for validation
            
        Returns:
            True if normalized correctly
        """
        if not self.is_fitted:
            logger.warning("Normalizer not fitted, cannot validate")
            return False
        
        if len(features) == 0:
            return False
        
        # Check mean and std for standardized features
        if self.method in ['standard', 'robust']:
            means = np.mean(features, axis=0)
            stds = np.std(features, axis=0)
            
            # Allow some tolerance
            mean_ok = np.all(np.abs(means) < tolerance)
            std_ok = np.all(np.abs(stds - 1) < tolerance)
            
            return mean_ok and std_ok
        
        # For minmax, check range
        elif self.method in ['minmax', 'maxabs']:
            min_vals = np.min(features, axis=0)
            max_vals = np.max(features, axis=0)
            
            if self.method == 'minmax':
                return np.all(min_vals >= 0 - tolerance) and np.all(max_vals <= 1 + tolerance)
            elif self.method == 'maxabs':
                return np.all(max_vals <= 1 + tolerance)
        
        return True
    
    def get_feature_stats(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics of normalized features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Statistics dictionary
        """
        if len(features) == 0:
            return {}
        
        return {
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'method': self.method,
            'is_fitted': self.is_fitted
        }


class FeaturePipeline:
    """
    Complete feature pipeline combining extraction and normalization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.extractor = FeatureExtractor(config)
        self.normalizer = FeatureNormalizer(config)
        self.is_fitted = False
        
        logger.info("FeaturePipeline initialized")
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self
        """
        # Extract features
        feature_matrix, _ = self.extractor.extract_all_features(df)
        
        if len(feature_matrix) == 0:
            logger.warning("No features extracted")
            return self
        
        # Fit normalizer
        self.normalizer.fit(feature_matrix)
        self.is_fitted = True
        
        logger.info(f"FeaturePipeline fitted on {feature_matrix.shape[0]} samples")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data through the pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Normalized feature matrix
        """
        # Extract features
        feature_matrix, _ = self.extractor.extract_all_features(df)
        
        if len(feature_matrix) == 0:
            logger.warning("No features extracted")
            return np.array([])
        
        # Normalize
        if self.is_fitted:
            return self.normalizer.transform(feature_matrix)
        else:
            return self.normalizer.fit_transform(feature_matrix)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Normalized feature matrix
        """
        return self.fit(df).transform(df)
    
    def save(self, path: str) -> None:
        """
        Save the pipeline.
        
        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'extractor': self.extractor,
                'normalizer': self.normalizer,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"FeaturePipeline saved to {path}")
    
    def load(self, path: str) -> 'FeaturePipeline':
        """
        Load the pipeline.
        
        Args:
            path: Path to load from
            
        Returns:
            Self
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.extractor = data['extractor']
        self.normalizer = data['normalizer']
        self.is_fitted = data['is_fitted']
        
        logger.info(f"FeaturePipeline loaded from {path}")
        return self


# Convenience functions
def normalize_features(features: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Quick function to normalize features.
    
    Args:
        features: Feature matrix
        config: Configuration dictionary
        
    Returns:
        Normalized features
    """
    normalizer = FeatureNormalizer(config)
    return normalizer.fit_transform(features)


def create_feature_pipeline(config: Dict[str, Any]) -> FeaturePipeline:
    """
    Quick function to create a feature pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FeaturePipeline instance
    """
    return FeaturePipeline(config)