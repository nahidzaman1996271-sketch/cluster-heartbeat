"""
Tests for data ingestion and preprocessing modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestion, ClusterMetrics
from src.data.preprocessing import DataPreprocessor
from src.data.synthetic_generator import SyntheticDataGenerator
from src.config import load_config


class TestDataIngestion:
    """Test data ingestion module."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def ingestion(self, config):
        """Create DataIngestion instance."""
        return DataIngestion(config)
    
    def test_initialization(self, ingestion):
        """Test DataIngestion initialization."""
        assert ingestion is not None
        assert ingestion.metrics is not None
        assert len(ingestion.metrics) > 0
        assert ingestion.window_size > 0
    
    def test_load_synthetic_data(self, ingestion):
        """Test loading synthetic data."""
        df = ingestion.load_data(source='synthetic')
        
        assert df is not None
        assert len(df) > 0
        assert 'gpu_utilization' in df.columns
        assert 'memory_utilization' in df.columns
        assert 'timestamp' in df.columns
        assert 'node_id' in df.columns
    
    def test_validate_data(self, ingestion):
        """Test data validation."""
        # Create valid data
        df = ingestion.load_data(source='synthetic')
        assert ingestion.validate_data(df) is True
        
        # Create invalid data (missing column)
        invalid_df = df.drop('gpu_utilization', axis=1)
        assert ingestion.validate_data(invalid_df) is False
    
    def test_data_stats(self, ingestion):
        """Test data statistics generation."""
        df = ingestion.load_data(source='synthetic')
        stats = ingestion.get_data_stats(df)
        
        assert 'total_rows' in stats
        assert 'total_columns' in stats
        assert 'unique_nodes' in stats
        assert stats['total_rows'] == len(df)
    
    def test_clear_cache(self, ingestion):
        """Test cache clearing."""
        # Load data to cache
        df = ingestion.load_data(source='synthetic')
        assert len(ingestion._cache) > 0
        
        ingestion.clear_cache()
        assert len(ingestion._cache) == 0


class TestDataPreprocessor:
    """Test data preprocessing module."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def preprocessor(self, config):
        """Create DataPreprocessor instance."""
        return DataPreprocessor(config)
    
    @pytest.fixture
    def sample_data(self, config):
        """Create sample data."""
        ingestion = DataIngestion(config)
        return ingestion.load_data(source='synthetic')
    
    def test_initialization(self, preprocessor):
        """Test DataPreprocessor initialization."""
        assert preprocessor is not None
        assert preprocessor.metrics is not None
        assert len(preprocessor.metrics) > 0
    
    def test_clean_data(self, preprocessor, sample_data):
        """Test data cleaning."""
        # Add some null values
        sample_data.loc[0:10, 'gpu_utilization'] = np.nan
        
        cleaned_df = preprocessor.clean_data(sample_data)
        
        assert cleaned_df is not None
        assert cleaned_df['gpu_utilization'].isnull().sum() == 0
        
        # Check outlier removal
        sample_data.loc[0, 'gpu_utilization'] = 100.0  # Extreme outlier
        cleaned_df = preprocessor.clean_data(sample_data)
        assert cleaned_df['gpu_utilization'].max() < 1.5
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit_transform method."""
        transformed_df = preprocessor.fit_transform(sample_data)
        
        assert transformed_df is not None
        assert 'gpu_utilization_scaled' in transformed_df.columns
        assert 'memory_utilization_scaled' in transformed_df.columns
    
    def test_get_feature_stats(self, preprocessor, sample_data):
        """Test feature statistics."""
        stats = preprocessor.get_feature_stats(sample_data)
        
        assert stats is not None
        assert 'gpu_utilization' in stats
        assert 'min' in stats['gpu_utilization']
        assert 'max' in stats['gpu_utilization']
        assert 'mean' in stats['gpu_utilization']
    
    def test_aggregate_time_series(self, preprocessor, sample_data):
        """Test time series aggregation."""
        # Add timestamp column
        sample_data['timestamp'] = np.arange(len(sample_data))
        
        aggregated = preprocessor.aggregate_time_series(
            sample_data,
            time_column='timestamp',
            interval='10s'
        )
        
        assert aggregated is not None
        assert len(aggregated) <= len(sample_data)


class TestSyntheticDataGenerator:
    """Test synthetic data generator."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def generator(self, config):
        """Create SyntheticDataGenerator instance."""
        return SyntheticDataGenerator(config)
    
    def test_initialization(self, generator):
        """Test SyntheticDataGenerator initialization."""
        assert generator is not None
        assert generator.num_nodes > 0
        assert generator.num_jobs > 0
        assert generator.time_steps > 0
    
    def test_generate(self, generator):
        """Test data generation."""
        metrics = generator.generate()
        
        assert metrics is not None
        assert isinstance(metrics, ClusterMetrics)
        
        # Check shapes
        df = metrics.to_dataframe()
        assert len(df) == generator.time_steps
        assert 'gpu_utilization' in df.columns
        assert 'memory_utilization' in df.columns
    
    def test_generate_node_data(self, generator):
        """Test node-specific data generation."""
        node_data = generator.generate_node_data(node_id=1, time_steps=100)
        
        assert node_data is not None
        assert len(node_data) == 100
        assert node_data['node_id'].iloc[0] == 1
    
    def test_get_metric_statistics(self, generator):
        """Test metric statistics."""
        stats = generator.get_metric_statistics()
        
        assert stats is not None
        assert 'gpu_utilization' in stats
        assert 'mean' in stats['gpu_utilization']
        assert stats['gpu_utilization']['mean'] >= 0
        assert stats['gpu_utilization']['mean'] <= 1
    
    def test_generate_batch(self, generator):
        """Test batch generation."""
        batches = generator.generate_batch(batch_size=50, num_batches=3)
        
        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 50
            assert 'gpu_utilization' in batch.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])