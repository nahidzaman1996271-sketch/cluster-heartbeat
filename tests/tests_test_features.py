"""
Tests for feature extraction and normalization modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.extractor import FeatureExtractor, WindowFeatures
from src.features.normalizer import FeatureNormalizer, FeaturePipeline
from src.data.ingestion import DataIngestion
from src.config import load_config


class TestFeatureExtractor:
    """Test feature extraction module."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def extractor(self, config):
        """Create FeatureExtractor instance."""
        return FeatureExtractor(config)
    
    @pytest.fixture
    def sample_data(self, config):
        """Create sample data."""
        ingestion = DataIngestion(config)
        return ingestion.load_data(source='synthetic')
    
    def test_initialization(self, extractor):
        """Test FeatureExtractor initialization."""
        assert extractor is not None
        assert extractor.metrics is not None
        assert extractor.window_size > 0
        assert extractor.stride > 0
    
    def test_extract_windows(self, extractor, sample_data):
        """Test window extraction."""
        windows = extractor.extract_windows(sample_data)
        
        assert len(windows) > 0
        for window in windows:
            assert window.shape[0] == extractor.window_size
            assert window.shape[1] == len(extractor.metrics)
    
    def test_extract_features(self, extractor):
        """Test feature extraction from a single window."""
        # Create a sample window
        window = np.random.rand(extractor.window_size, len(extractor.metrics))
        
        features = extractor.extract_features(window)
        
        assert isinstance(features, WindowFeatures)
        assert len(features) > 0
        
        # Check feature types
        assert len(features.statistical) > 0
        assert len(features.trend) > 0
        assert len(features.spectral) > 0
        
        # Test flatten
        flat_features = features.flatten()
        assert len(flat_features) > 0
    
    def test_extract_all_features(self, extractor, sample_data):
        """Test extracting features from all windows."""
        feature_matrix, window_features = extractor.extract_all_features(sample_data)
        
        assert len(feature_matrix) > 0
        assert len(window_features) > 0
        assert len(feature_matrix) == len(window_features)
    
    def test_extract_for_job(self, extractor, sample_data):
        """Test extracting features for a specific job."""
        if 'job_id' in sample_data.columns:
            job_id = sample_data['job_id'].iloc[0]
            features = extractor.extract_for_job(sample_data, job_id)
            
            assert len(features) > 0
    
    def test_extract_for_node(self, extractor, sample_data):
        """Test extracting features for a specific node."""
        if 'node_id' in sample_data.columns:
            node_id = sample_data['node_id'].iloc[0]
            features = extractor.extract_for_node(sample_data, node_id)
            
            assert len(features) > 0
    
    def test_get_feature_stats(self, extractor, sample_data):
        """Test feature statistics."""
        feature_matrix, _ = extractor.extract_all_features(sample_data)
        
        if len(feature_matrix) > 0:
            stats = extractor.get_feature_stats(feature_matrix)
            assert stats is not None
            assert 'n_samples' in stats
            assert 'n_features' in stats


class TestFeatureNormalizer:
    """Test feature normalization module."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def normalizer(self, config):
        """Create FeatureNormalizer instance."""
        return FeatureNormalizer(config)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        return np.random.randn(100, 32)
    
    def test_initialization(self, normalizer):
        """Test FeatureNormalizer initialization."""
        assert normalizer is not None
        assert normalizer.method == 'standard'
        assert normalizer.is_fitted is False
    
    def test_fit_transform(self, normalizer, sample_features):
        """Test fit_transform method."""
        transformed = normalizer.fit_transform(sample_features)
        
        assert transformed is not None
        assert transformed.shape == sample_features.shape
        assert normalizer.is_fitted is True
        
        # Check normalization (mean ~0, std ~1)
        assert np.abs(np.mean(transformed)) < 1e-6
        assert np.abs(np.std(transformed) - 1) < 1e-6
    
    def test_transform(self, normalizer, sample_features):
        """Test transform after fitting."""
        normalizer.fit(sample_features)
        transformed = normalizer.transform(sample_features)
        
        assert transformed is not None
        assert transformed.shape == sample_features.shape
    
    def test_inverse_transform(self, normalizer, sample_features):
        """Test inverse transformation."""
        normalizer.fit(sample_features)
        transformed = normalizer.transform(sample_features)
        reconstructed = normalizer.inverse_transform(transformed)
        
        assert np.allclose(reconstructed, sample_features, rtol=1e-5)
    
    def test_save_load(self, normalizer, sample_features, tmp_path):
        """Test save and load functionality."""
        normalizer.fit(sample_features)
        
        # Save
        save_path = tmp_path / 'normalizer.pkl'
        normalizer.save(str(save_path))
        
        # Load
        new_normalizer = FeatureNormalizer(self.config)
        new_normalizer.load(str(save_path))
        
        assert new_normalizer.is_fitted is True
        assert new_normalizer.method == normalizer.method
        
        # Test transform after load
        transformed = new_normalizer.transform(sample_features)
        assert transformed.shape == sample_features.shape


class TestFeaturePipeline:
    """Test complete feature pipeline."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def pipeline(self, config):
        """Create FeaturePipeline instance."""
        return FeaturePipeline(config)
    
    @pytest.fixture
    def sample_data(self, config):
        """Create sample data."""
        ingestion = DataIngestion(config)
        return ingestion.load_data(source='synthetic')
    
    def test_initialization(self, pipeline):
        """Test FeaturePipeline initialization."""
        assert pipeline is not None
        assert pipeline.extractor is not None
        assert pipeline.normalizer is not None
    
    def test_fit_transform(self, pipeline, sample_data):
        """Test fit_transform method."""
        features = pipeline.fit_transform(sample_data)
        
        assert features is not None
        assert len(features) > 0
        assert pipeline.is_fitted is True
    
    def test_transform(self, pipeline, sample_data):
        """Test transform after fitting."""
        pipeline.fit(sample_data)
        features = pipeline.transform(sample_data)
        
        assert features is not None
        assert len(features) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])