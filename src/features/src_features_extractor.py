"""
Feature extraction module for Cluster Heartbeat.
Extracts sliding window features from cluster metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction.
    """
    statistical: bool = True
    statistical_features: List[str] = field(default_factory=lambda: [
        'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'skew', 'kurtosis'
    ])
    
    trend: bool = True
    trend_features: List[str] = field(default_factory=lambda: [
        'slope', 'change_rate', 'acceleration'
    ])
    
    spectral: bool = True
    spectral_components: int = 5
    spectral_features: List[str] = field(default_factory=lambda: [
        'dominant_frequency', 'spectral_centroid', 'spectral_spread'
    ])
    
    interaction: bool = True
    interaction_features: List[str] = field(default_factory=lambda: [
        'correlations', 'cross_metrics'
    ])
    
    temporal: bool = True
    temporal_features: List[str] = field(default_factory=lambda: [
        'autocorrelation', 'zero_crossing_rate'
    ])
    
    window_size: int = 300
    stride: int = 60


@dataclass
class WindowFeatures:
    """
    Container for window-based features.
    """
    statistical: Dict[str, np.ndarray] = field(default_factory=dict)
    trend: Dict[str, np.ndarray] = field(default_factory=dict)
    spectral: Dict[str, np.ndarray] = field(default_factory=dict)
    interaction: Dict[str, np.ndarray] = field(default_factory=dict)
    temporal: Dict[str, np.ndarray] = field(default_factory=dict)
    raw: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def flatten(self) -> np.ndarray:
        """
        Flatten all features into a single vector.
        
        Returns:
            Flattened feature vector
        """
        features = []
        feature_groups = [self.statistical, self.trend, self.spectral, 
                         self.interaction, self.temporal]
        
        for group in feature_groups:
            for key, value in group.items():
                if value is not None and value.size > 0:
                    # Ensure value is 1D
                    if value.ndim > 1:
                        value = value.flatten()
                    features.append(value)
        
        # Add raw data if available
        if self.raw is not None:
            features.append(self.raw.flatten())
        
        # Combine all features
        if features:
            return np.concatenate([f for f in features if f.size > 0])
        else:
            return np.array([])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'statistical': {k: v.tolist() for k, v in self.statistical.items() if v is not None},
            'trend': {k: v.tolist() for k, v in self.trend.items() if v is not None},
            'spectral': {k: v.tolist() for k, v in self.spectral.items() if v is not None},
            'interaction': {k: v.tolist() for k, v in self.interaction.items() if v is not None},
            'temporal': {k: v.tolist() for k, v in self.temporal.items() if v is not None},
            'metadata': self.metadata
        }
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features.
        
        Returns:
            List of feature names
        """
        names = []
        for group_name, group in [
            ('statistical', self.statistical),
            ('trend', self.trend),
            ('spectral', self.spectral),
            ('interaction', self.interaction),
            ('temporal', self.temporal)
        ]:
            for key in group.keys():
                names.append(f"{group_name}_{key}")
        return names
    
    def __len__(self) -> int:
        """Get total number of features."""
        return len(self.flatten())


class FeatureExtractor:
    """
    Extracts features from sliding windows of cluster metrics.
    Creates comprehensive feature sets for fingerprint generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Feature configuration
        feat_config = config['features']
        self.metrics = feat_config['metrics']
        self.window_size = config['data']['processing']['window_size']
        self.stride = config['data']['processing']['stride']
        self.embedding_dim = feat_config.get('embedding_dim', 32)
        
        # Feature extraction settings
        extraction_config = feat_config.get('extraction', {})
        self.feature_config = FeatureConfig(
            statistical=True,
            trend=True,
            spectral=True,
            interaction=True,
            temporal=True,
            window_size=self.window_size,
            stride=self.stride
        )
        
        # PCA for feature reduction (optional)
        self.pca = None
        self.use_pca = feat_config.get('reduction', {}).get('enabled', False)
        if self.use_pca:
            self.pca_components = feat_config.get('reduction', {}).get('components', 0.95)
        
        logger.info(f"FeatureExtractor initialized with {len(self.metrics)} metrics")
    
    def extract_windows(self, df: pd.DataFrame) -> List[np.ndarray]:
        """
        Extract sliding windows from data.
        
        Args:
            df: DataFrame with cluster metrics
            
        Returns:
            List of window arrays
        """
        windows = []
        total_rows = len(df)
        
        # Get numeric columns for window extraction
        window_cols = [col for col in self.metrics if col in df.columns]
        
        if not window_cols:
            logger.warning("No metric columns found in DataFrame")
            return windows
        
        for start in range(0, total_rows - self.window_size, self.stride):
            end = start + self.window_size
            window_data = df[window_cols].iloc[start:end].values
            windows.append(window_data)
        
        logger.info(f"Extracted {len(windows)} windows of size {self.window_size}")
        return windows
    
    def extract_features(self, window: np.ndarray) -> WindowFeatures:
        """
        Extract comprehensive features from a single window.
        
        Args:
            window: 2D array of shape (window_size, n_metrics)
            
        Returns:
            WindowFeatures object
        """
        n_metrics = window.shape[1]
        window_size = window.shape[0]
        
        # Initialize feature dictionaries
        statistical = {}
        trend = {}
        spectral = {}
        interaction = {}
        temporal = {}
        
        # 1. Statistical features
        if self.feature_config.statistical:
            for i in range(n_metrics):
                col_data = window[:, i]
                stats_dict = {
                    'mean': np.mean(col_data),
                    'std': np.std(col_data),
                    'min': np.min(col_data),
                    'max': np.max(col_data),
                    'median': np.median(col_data),
                    'q25': np.percentile(col_data, 25),
                    'q75': np.percentile(col_data, 75),
                    'skew': stats.skew(col_data) if len(col_data) > 2 else 0,
                    'kurtosis': stats.kurtosis(col_data) if len(col_data) > 3 else 0
                }
                
                # Add range and iqr
                stats_dict['range'] = stats_dict['max'] - stats_dict['min']
                stats_dict['iqr'] = stats_dict['q75'] - stats_dict['q25']
                
                # Add coefficient of variation
                if stats_dict['mean'] != 0:
                    stats_dict['cv'] = stats_dict['std'] / abs(stats_dict['mean'])
                else:
                    stats_dict['cv'] = 0
                
                # Add to statistical features
                for key, value in stats_dict.items():
                    statistical[f'col_{i}_{key}'] = np.array([value])
        
        # 2. Trend features
        if self.feature_config.trend:
            for i in range(n_metrics):
                col_data = window[:, i]
                x = np.arange(len(col_data))
                
                # Linear trend
                slope, intercept = np.polyfit(x, col_data, 1)
                trend[f'col_{i}_slope'] = np.array([slope])
                trend[f'col_{i}_intercept'] = np.array([intercept])
                
                # Change rate
                if len(col_data) > 1:
                    change_rate = (col_data[-1] - col_data[0]) / len(col_data)
                    trend[f'col_{i}_change_rate'] = np.array([change_rate])
                
                # Acceleration (second derivative)
                if len(col_data) > 2:
                    coeffs = np.polyfit(x, col_data, 2)
                    acceleration = 2 * coeffs[0]  # Second derivative coefficient
                    trend[f'col_{i}_acceleration'] = np.array([acceleration])
        
        # 3. Spectral features (FFT)
        if self.feature_config.spectral:
            for i in range(n_metrics):
                col_data = window[:, i]
                
                # FFT
                fft_vals = fft(col_data)
                freqs = fftfreq(len(col_data))
                magnitude = np.abs(fft_vals)
                
                # Dominant frequency
                if len(magnitude) > 1:
                    dominant_idx = np.argmax(magnitude[1:]) + 1
                    dominant_freq = freqs[dominant_idx]
                    spectral[f'col_{i}_dominant_freq'] = np.array([dominant_freq])
                
                # Top spectral components
                n_components = min(self.feature_config.spectral_components, len(magnitude) // 2)
                if n_components > 0:
                    # Get top frequencies
                    top_indices = np.argsort(magnitude[1:])[-n_components:] + 1
                    for j, idx in enumerate(top_indices):
                        spectral[f'col_{i}_fft_{j}'] = np.array([magnitude[idx]])
                        spectral[f'col_{i}_freq_{j}'] = np.array([freqs[idx]])
                
                # Spectral centroid
                if np.sum(magnitude) > 0:
                    spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                    spectral[f'col_{i}_spectral_centroid'] = np.array([spectral_centroid])
        
        # 4. Interaction features (between metrics)
        if self.feature_config.interaction and n_metrics > 1:
            # Correlation matrix
            corr_matrix = np.corrcoef(window.T)
            
            # Get upper triangle indices (excluding diagonal)
            upper_indices = np.triu_indices_from(corr_matrix, k=1)
            
            # Store correlations
            for idx, (i, j) in enumerate(zip(upper_indices[0], upper_indices[1])):
                interaction[f'corr_{i}_{j}'] = np.array([corr_matrix[i, j]])
            
            # Cross-metric ratios
            for i in range(n_metrics):
                for j in range(i+1, n_metrics):
                    # Avoid division by zero
                    mean_i = np.mean(window[:, i])
                    mean_j = np.mean(window[:, j])
                    if mean_j != 0:
                        ratio = mean_i / mean_j
                        interaction[f'ratio_{i}_{j}'] = np.array([ratio])
        
        # 5. Temporal features
        if self.feature_config.temporal:
            for i in range(n_metrics):
                col_data = window[:, i]
                
                # Autocorrelation at lag 1
                if len(col_data) > 1:
                    autocorr = np.corrcoef(col_data[:-1], col_data[1:])[0, 1]
                    temporal[f'col_{i}_autocorr_lag1'] = np.array([autocorr])
                
                # Zero crossing rate
                if len(col_data) > 1:
                    # Center data
                    centered = col_data - np.mean(col_data)
                    zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
                    zero_crossing_rate = zero_crossings / len(col_data)
                    temporal[f'col_{i}_zero_crossing_rate'] = np.array([zero_crossing_rate])
                
                # Entropy (approximate)
                if len(col_data) > 1:
                    hist, _ = np.histogram(col_data, bins='auto')
                    hist = hist / np.sum(hist)
                    hist = hist[hist > 0]  # Remove zeros for entropy
                    if len(hist) > 0:
                        entropy = -np.sum(hist * np.log2(hist))
                        temporal[f'col_{i}_entropy'] = np.array([entropy])
        
        return WindowFeatures(
            statistical=statistical,
            trend=trend,
            spectral=spectral,
            interaction=interaction,
            temporal=temporal,
            raw=window,
            metadata={
                'window_size': window_size,
                'n_metrics': n_metrics,
                'timestamp': None  # Can be set later
            }
        )
    
    def extract_all_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[WindowFeatures]]:
        """
        Extract features from all windows.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (feature_matrix, list of WindowFeatures objects)
        """
        windows = self.extract_windows(df)
        window_features = []
        feature_matrix = []
        
        for i, window in enumerate(windows):
            features = self.extract_features(window)
            window_features.append(features)
            
            # Flatten features
            feature_vector = features.flatten()
            feature_matrix.append(feature_vector)
            
            # Add window index to metadata
            features.metadata['window_index'] = i
        
        if feature_matrix:
            feature_matrix = np.array(feature_matrix)
            logger.info(f"Extracted {len(feature_matrix)} feature vectors of length {feature_matrix.shape[1]}")
        else:
            feature_matrix = np.array([])
            logger.warning("No features extracted")
        
        # Apply PCA if enabled
        if self.use_pca and len(feature_matrix) > 0:
            feature_matrix = self._apply_pca(feature_matrix)
        
        return feature_matrix, window_features
    
    def _apply_pca(self, features: np.ndarray) -> np.ndarray:
        """
        Apply PCA for feature reduction.
        
        Args:
            features: Feature matrix
            
        Returns:
            Reduced feature matrix
        """
        if self.pca is None:
            self.pca = PCA(n_components=self.pca_components)
            reduced = self.pca.fit_transform(features)
            logger.info(f"PCA reduced features from {features.shape[1]} to {reduced.shape[1]} dimensions")
        else:
            reduced = self.pca.transform(features)
        
        return reduced
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance if PCA is used.
        
        Returns:
            Feature importance dictionary
        """
        if self.pca is None:
            return {}
        
        importance = {
            f'component_{i}': float(var)
            for i, var in enumerate(self.pca.explained_variance_ratio_)
        }
        return importance
    
    def extract_for_job(self, df: pd.DataFrame, job_id: Union[str, int]) -> np.ndarray:
        """
        Extract features for a specific job.
        
        Args:
            df: Input DataFrame
            job_id: Job identifier
            
        Returns:
            Feature matrix for the job
        """
        if 'job_id' not in df.columns:
            logger.warning("No job_id column found")
            return np.array([])
        
        job_df = df[df['job_id'] == job_id]
        
        if len(job_df) == 0:
            logger.warning(f"Job {job_id} not found")
            return np.array([])
        
        features, _ = self.extract_all_features(job_df)
        return features
    
    def extract_for_node(self, df: pd.DataFrame, node_id: Union[str, int]) -> np.ndarray:
        """
        Extract features for a specific node.
        
        Args:
            df: Input DataFrame
            node_id: Node identifier
            
        Returns:
            Feature matrix for the node
        """
        if 'node_id' not in df.columns:
            logger.warning("No node_id column found")
            return np.array([])
        
        node_df = df[df['node_id'] == node_id]
        
        if len(node_df) == 0:
            logger.warning(f"Node {node_id} not found")
            return np.array([])
        
        features, _ = self.extract_all_features(node_df)
        return features
    
    def extract_latest_features(self, df: pd.DataFrame, window_size: Optional[int] = None) -> WindowFeatures:
        """
        Extract features from the latest window of data.
        
        Args:
            df: Input DataFrame
            window_size: Size of window (uses default if None)
            
        Returns:
            WindowFeatures for the latest window
        """
        if window_size is None:
            window_size = self.window_size
        
        if len(df) < window_size:
            logger.warning(f"Not enough data for window: {len(df)} < {window_size}")
            return WindowFeatures()
        
        # Take the last window_size rows
        latest_data = df.iloc[-window_size:].values
        return self.extract_features(latest_data)
    
    def get_feature_stats(self, feature_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics of extracted features.
        
        Args:
            feature_matrix: Feature matrix
            
        Returns:
            Statistics dictionary
        """
        if len(feature_matrix) == 0:
            return {}
        
        return {
            'n_samples': feature_matrix.shape[0],
            'n_features': feature_matrix.shape[1],
            'mean': float(np.mean(feature_matrix)),
            'std': float(np.std(feature_matrix)),
            'min': float(np.min(feature_matrix)),
            'max': float(np.max(feature_matrix)),
            'variance_explained': self.pca.explained_variance_ratio_.tolist() if self.pca else None
        }


# Convenience functions
def extract_features(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, List[WindowFeatures]]:
    """
    Quick function to extract features.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (feature_matrix, list of WindowFeatures)
    """
    extractor = FeatureExtractor(config)
    return extractor.extract_all_features(df)