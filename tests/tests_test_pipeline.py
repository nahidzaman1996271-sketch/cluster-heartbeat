"""
Tests for pipeline and service integration.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipeline import ClusterHeartbeatPipeline
from src.core.service import ClusterHeartbeatService
from src.config import load_config
from src.data.ingestion import DataIngestion


class TestPipeline:
    """Test pipeline functionality."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def pipeline(self, config):
        """Create pipeline instance."""
        return ClusterHeartbeatPipeline(config)
    
    @pytest.fixture
    def sample_data(self, config):
        """Create sample data."""
        ingestion = DataIngestion(config)
        return ingestion.load_data(source='synthetic')
    
    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert pipeline.is_initialized is False
        
        pipeline.initialize(load_models=False)
        assert pipeline.is_initialized is True
    
    def test_process_batch(self, pipeline, sample_data):
        """Test batch processing."""
        pipeline.initialize(load_models=False)
        
        results = pipeline.process_batch(sample_data)
        
        assert results is not None
        assert 'timestamp' in results
        assert 'summary' in results
        assert 'fingerprints' in results
        assert 'health_scores' in results
        assert 'pipeline_stats' in results
    
    def test_process_single_node(self, pipeline, sample_data):
        """Test single node processing."""
        pipeline.initialize(load_models=False)
        
        # First process batch to cache data
        pipeline.process_batch(sample_data)
        
        if 'node_id' in sample_data.columns:
            node_id = sample_data['node_id'].iloc[0]
            results = pipeline.process_single_node(str(node_id))
            
            assert results is not None
            assert 'node_id' in results or 'error' in results
    
    def test_get_dashboard_data(self, pipeline, sample_data):
        """Test dashboard data generation."""
        pipeline.initialize(load_models=False)
        
        # Process data first
        pipeline.process_batch(sample_data)
        
        dashboard = pipeline.get_dashboard_data()
        
        assert dashboard is not None
        assert 'cluster_summary' in dashboard
        assert 'health_scores' in dashboard
        assert 'anomalies' in dashboard
    
    def test_pipeline_stats(self, pipeline, sample_data):
        """Test pipeline statistics."""
        pipeline.initialize(load_models=False)
        
        # Get initial stats
        stats = pipeline.get_pipeline_stats()
        assert stats['total_processed'] == 0
        
        # Process data
        pipeline.process_batch(sample_data)
        
        # Get updated stats
        stats = pipeline.get_pipeline_stats()
        assert stats['total_processed'] > 0
        assert 'avg_processing_time' in stats
        assert stats['is_initialized'] is True
    
    def test_save_models(self, pipeline, sample_data, tmp_path):
        """Test model saving."""
        pipeline.initialize(load_models=False)
        pipeline.model_dir = tmp_path
        
        # Process data to train models
        pipeline.process_batch(sample_data)
        
        # Save models
        pipeline.save_models()
        
        # Check files
        assert (tmp_path / 'normalizer.pkl').exists()
    
    def test_shutdown(self, pipeline):
        """Test pipeline shutdown."""
        pipeline.initialize(load_models=False)
        assert pipeline.is_initialized is True
        
        pipeline.shutdown()
        assert pipeline.is_initialized is False


class TestService:
    """Test service functionality."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def service(self, config):
        """Create service instance."""
        service = ClusterHeartbeatService(config)
        service.start()
        return service
    
    @pytest.fixture
    def sample_data(self, config):
        """Create sample data."""
        ingestion = DataIngestion(config)
        return ingestion.load_data(source='synthetic')
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert service.is_running is True
        assert service.health_status == "running"
    
    def test_process_metrics(self, service, sample_data):
        """Test metrics processing."""
        results = service.process_metrics(sample_data.to_dict('records'))
        
        assert results is not None
        assert 'timestamp' in results
        assert 'summary' in results
    
    def test_async_processing(self, service, sample_data):
        """Test async processing."""
        job_id = service.process_async(sample_data.to_dict('records'))
        
        assert job_id is not None
        assert job_id.startswith('job_')
        
        # Wait for processing
        time.sleep(1)
        
        status = service.get_job_status(job_id)
        assert status['job_id'] == job_id
        assert status['status'] in ['queued', 'processing', 'completed']
    
    def test_get_cluster_status(self, service, sample_data):
        """Test cluster status."""
        # Process data first
        service.process_metrics(sample_data.to_dict('records'))
        
        status = service.get_cluster_status()
        
        assert status is not None
        assert 'status' in status
        assert 'timestamp' in status
        assert 'service_stats' in status
    
    def test_get_health_score(self, service, sample_data):
        """Test health score retrieval."""
        service.process_metrics(sample_data.to_dict('records'))
        
        health = service.get_health_score()
        assert health is not None
        
        # Test node-specific
        if 'node_id' in sample_data.columns:
            node_id = str(sample_data['node_id'].iloc[0])
            node_health = service.get_health_score(node_id)
            assert node_health is not None
    
    def test_get_predictions(self, service, sample_data):
        """Test predictions retrieval."""
        service.process_metrics(sample_data.to_dict('records'))
        
        predictions = service.get_predictions()
        
        assert predictions is not None
        assert 'predictions' in predictions
    
    def test_get_scheduling(self, service, sample_data):
        """Test scheduling recommendations."""
        service.process_metrics(sample_data.to_dict('records'))
        
        scheduling = service.get_scheduling_recommendations()
        
        assert scheduling is not None
        assert 'recommendations' in scheduling
    
    def test_get_cost_savings(self, service, sample_data):
        """Test cost savings retrieval."""
        service.process_metrics(sample_data.to_dict('records'))
        
        cost = service.get_cost_savings()
        
        assert cost is not None
        assert 'savings' in cost
    
    def test_get_dashboard(self, service, sample_data):
        """Test dashboard data."""
        service.process_metrics(sample_data.to_dict('records'))
        
        dashboard = service.get_dashboard_data()
        
        assert dashboard is not None
        assert 'cluster_summary' in dashboard
    
    def test_get_service_stats(self, service, sample_data):
        """Test service statistics."""
        service.process_metrics(sample_data.to_dict('records'))
        
        stats = service.get_service_stats()
        
        assert stats is not None
        assert 'uptime_seconds' in stats
        assert 'total_jobs' in stats
        assert 'processed_jobs' in stats
    
    def test_clear_cache(self, service, sample_data):
        """Test cache clearing."""
        service.process_metrics(sample_data.to_dict('records'))
        assert len(service.cache) > 0
        
        service.clear_cache()
        assert len(service.cache) == 0
    
    def test_stop_service(self, service):
        """Test service shutdown."""
        assert service.is_running is True
        
        service.stop()
        assert service.is_running is False
        assert service.health_status == "stopped"
    
    def test_job_status(self, service, sample_data):
        """Test job status retrieval."""
        job_id = service.process_async(sample_data.to_dict('records'))
        
        # Check immediately (should be queued or processing)
        status = service.get_job_status(job_id)
        assert status['job_id'] == job_id
        assert status['status'] in ['queued', 'processing']
        
        # Wait and check again
        time.sleep(2)
        status = service.get_job_status(job_id)
        assert status['job_id'] == job_id


class TestIntegration:
    """Integration tests for the full system."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def service(self, config):
        """Create service instance."""
        service = ClusterHeartbeatService(config)
        service.start()
        return service
    
    def test_end_to_end(self, service, config):
        """Test end-to-end flow."""
        # 1. Load data
        ingestion = DataIngestion(config)
        df = ingestion.load_data(source='synthetic')
        
        # 2. Process data
        results = service.process_metrics(df.to_dict('records'))
        assert results is not None
        
        # 3. Get predictions
        predictions = service.get_predictions()
        assert 'predictions' in predictions
        
        # 4. Get recommendations
        scheduling = service.get_scheduling_recommendations()
        cost = service.get_cost_savings()
        
        # 5. Get dashboard
        dashboard = service.get_dashboard_data()
        
        # 6. Verify all components work together
        assert 'cluster_summary' in dashboard
        assert 'predictions' in dashboard
        assert 'scheduling' in dashboard
        assert 'cost_savings' in dashboard
    
    def test_continuous_processing(self, service, config):
        """Test continuous data processing."""
        ingestion = DataIngestion(config)
        
        # Process multiple batches
        for i in range(3):
            df = ingestion.load_data(source='synthetic')
            results = service.process_metrics(df.to_dict('records'))
            assert results is not None
            
            # Check service stats
            stats = service.get_service_stats()
            assert stats['processed_jobs'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])