"""
Data preprocessing module for Cluster Heartbeat.
Handles cleaning, normalization, and transformation of cluster metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
from pathlib import Path
import pickle

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """
    Configuration for data preprocessing.
    """
    handle_missing: str = 'mean'  # mean, median, most_frequent, constant, drop
    scaling_method: str = 'standard'  # standard, robust, minmax, none
    remove_outliers: bool = True
    outlier_method: str = 'iqr'  # iqr, zscore
    outlier_threshold: float = 3.0
    normalize_features: bool = True
    feature_selection: Optional[List[str]] = None
    time_aggregation: Optional[str] = None  # mean, sum, max, min


class DataPreprocessor:
    """
    Preprocesses cluster metrics data.
    Handles cleaning, imputation, scaling, and transformation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics = config['features']['metrics']
        self.normalization_config = config['features']['normalization']
        
        # Initialize transformers
        self.scaler = None
        self.imputer = None
        self.is_fitted = False
        
        # State
        self.feature_names = None
        self.scaler_type = self.normalization_config.get('method', 'standard')
        
        logger.info(f"DataPreprocessor initialized with {len(self.metrics)} metrics")
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Self
        """
        # Select features
        feature_cols = [col for col in self.metrics if col in df.columns]
        
        if not feature_cols:
            logger.warning("No feature columns found in DataFrame")
            return self
        
        # Extract feature matrix
        X = df[feature_cols].values
        
        # Fit imputer
        imputer_strategy = self.normalization_config.get('method', 'mean')
        if imputer_strategy in ['mean', 'median', 'most_frequent', 'constant']:
            self.imputer = SimpleImputer(strategy=imputer_strategy)
            self.imputer.fit(X)
        else:
            self.imputer = SimpleImputer(strategy='mean')
            self.imputer.fit(X)
        
        # Fit scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        # Apply imputation before scaling
        X_imputed = self.imputer.transform(X)
        self.scaler.fit(X_imputed)
        
        self.feature_names = feature_cols
        self.is_fitted = True
        
        logger.info(f"Preprocessor fitted on {len(feature_cols)} features")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Select features
        feature_cols = [col for col in self.metrics if col in df.columns]
        
        if not feature_cols:
            logger.warning("No feature columns found in DataFrame")
            return df
        
        # Extract feature matrix
        X = df[feature_cols].values
        
        # Apply imputation
        X_imputed = self.imputer.transform(X)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X_imputed)
        
        # Create transformed DataFrame
        transformed_df = df.copy()
        for i, col in enumerate(feature_cols):
            transformed_df[f'{col}_scaled'] = X_scaled[:, i]
        
        return transformed_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data.
        
        Args:
            df: Scaled DataFrame
            
        Returns:
            Original scale DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        feature_cols = [col for col in df.columns if col.endswith('_scaled')]
        original_cols = [col.replace('_scaled', '') for col in feature_cols]
        
        if not feature_cols:
            return df
        
        # Extract scaled features
        X_scaled = df[feature_cols].values
        
        # Inverse scale
        X_original = self.scaler.inverse_transform(X_scaled)
        
        # Create DataFrame
        transformed_df = df.copy()
        for i, col in enumerate(original_cols):
            transformed_df[col] = X_original[:, i]
        
        return transformed_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values
        for col in self.metrics:
            if col in df_clean.columns:
                if df_clean[col].isnull().any():
                    # Use median for numeric columns
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    logger.debug(f"Filled missing values in {col} with median: {median_val:.4f}")
        
        # Remove outliers
        if self.normalization_config.get('scaling', True):
            for col in self.metrics:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing them
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Cleaned data: {len(df_clean)} rows")
        return df_clean
    
    def aggregate_time_series(self, df: pd.DataFrame, 
                            time_column: str = 'timestamp', 
                            interval: str = '1min') -> pd.DataFrame:
        """
        Aggregate time series data by time interval.
        
        Args:
            df: Input DataFrame
            time_column: Name of timestamp column
            interval: Aggregation interval (e.g., '1min', '5min', '1h')
            
        Returns:
            Aggregated DataFrame
        """
        df_agg = df.copy()
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_agg[time_column]):
            df_agg[time_column] = pd.to_datetime(df_agg[time_column], unit='s')
        
        # Set index and resample
        df_agg.set_index(time_column, inplace=True)
        
        # Aggregate numeric columns
        numeric_cols = [col for col in df_agg.columns if col != 'node_id' and col != 'job_id']
        df_resampled = df_agg[numeric_cols].resample(interval).mean()
        
        # Reset index
        df_resampled.reset_index(inplace=True)
        
        # Add node_id and job_id if they exist
        if 'node_id' in df_agg.columns:
            df_resampled['node_id'] = df_agg['node_id'].mode().iloc[0] if not df_agg['node_id'].empty else 'unknown'
        if 'job_id' in df_agg.columns:
            df_resampled['job_id'] = df_agg['job_id'].mode().iloc[0] if not df_agg['job_id'].empty else 'unknown'
        
        logger.info(f"Aggregated data to {interval} intervals: {len(df_resampled)} rows")
        return df_resampled
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using fitted scaler.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Normalized DataFrame
        """
        if not self.is_fitted:
            return self.fit_transform(df)
        return self.transform(df)
    
    def get_feature_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics for features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        for col in self.metrics:
            if col in df.columns:
                stats[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'skew': float(df[col].skew()),
                    'kurtosis': float(df[col].kurtosis()),
                    'null_count': int(df[col].isnull().sum()),
                    'null_percentage': float(df[col].isnull().sum() / len(df) * 100)
                }
        
        return stats
    
    def save(self, path: str) -> None:
        """
        Save the preprocessor to disk.
        
        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'imputer': self.imputer,
                'feature_names': self.feature_names,
                'scaler_type': self.scaler_type,
                'is_fitted': self.is_fitted,
                'metrics': self.metrics
            }, f)
        
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str) -> 'DataPreprocessor':
        """
        Load the preprocessor from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            Self
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.scaler = data['scaler']
        self.imputer = data['imputer']
        self.feature_names = data['feature_names']
        self.scaler_type = data['scaler_type']
        self.is_fitted = data['is_fitted']
        self.metrics = data['metrics']
        
        logger.info(f"Preprocessor loaded from {path}")
        return self
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df_new = df.copy()
        
        # Create interaction features
        for i, col1 in enumerate(self.metrics):
            if col1 not in df.columns:
                continue
            for col2 in self.metrics[i+1:]:
                if col2 not in df.columns:
                    continue
                feature_name = f'{col1}_{col2}_interaction'
                df_new[feature_name] = df[col1] * df[col2]
        
        # Create ratio features
        for col in self.metrics:
            if col not in df.columns:
                continue
            # GPU utilization ratio
            if 'gpu' in col and 'memory' in col:
                # Create ratio if both exist
                pass
        
        logger.info(f"Created additional features: {len(df_new.columns) - len(df.columns)} new columns")
        return df_new
    
    def detect_anomalies(self, df: pd.DataFrame, method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Args:
            df: Input DataFrame
            method: Detection method ('zscore', 'iqr')
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly flags
        """
        df_anomaly = df.copy()
        anomaly_flags = pd.DataFrame(index=df.index)
        
        for col in self.metrics:
            if col not in df.columns:
                continue
            
            if method == 'zscore':
                zscore = np.abs((df[col] - df[col].mean()) / df[col].std())
                anomaly_flags[f'{col}_anomaly'] = zscore > threshold
                
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomaly_flags[f'{col}_anomaly'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        # Overall anomaly flag
        df_anomaly['is_anomaly'] = anomaly_flags.any(axis=1)
        df_anomaly['anomaly_count'] = anomaly_flags.sum(axis=1)
        
        logger.info(f"Detected {df_anomaly['is_anomaly'].sum()} anomalous rows")
        return df_anomaly


# Convenience functions
def preprocess_cluster_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Quick function to preprocess cluster data.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.fit_transform(df)


def clean_cluster_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Quick function to clean cluster data.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Cleaned DataFrame
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.clean_data(df)