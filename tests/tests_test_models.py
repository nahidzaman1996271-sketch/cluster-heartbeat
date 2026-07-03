"""
Tests for model modules.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fingerprint import FingerprintAutoencoder, FingerprintTrainer
from src.models.anomaly import AnomalyDetector
from src.models.scheduler import SmartScheduler, SchedulingRecommendation
from src.models.cost_optimizer import CostOptimizer, IdleGPUInfo
from src.config import load_config


class TestFingerprintAutoencoder:
    """Test fingerprint autoencoder."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def model(self):
        """Create autoencoder model."""
        return FingerprintAutoencoder(
            input_dim=128,
            latent_dim=32,
            hidden_dims=[64, 128, 64]
        )
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.input_dim == 128
        assert model.latent_dim == 32
        assert len(model.hidden_dims) == 3
    
    def test_forward(self, model):
        """Test forward pass."""
        x = torch.randn(16, 128)
        reconstructed, latent = model(x)
        
        assert reconstructed.shape == x.shape
        assert latent.shape == (16, 32)
    
    def test_encode_decode(self, model):
        """Test encode and decode methods."""
        x = torch.randn(16, 128)
        latent = model.encode(x)
        reconstructed = model.decode(latent)
        
        assert latent.shape == (16, 32)
        assert reconstructed.shape == x.shape
    
    def test_parameter_count(self, model):
        """Test parameter count."""
        count = model.get_parameter_count()
        assert count > 0


class TestFingerprintTrainer:
    """Test fingerprint trainer."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def trainer(self, config):
        """Create FingerprintTrainer instance."""
        return FingerprintTrainer(config)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        return np.random.randn(200, 128)
    
    def test_initialization(self, trainer):
        """Test FingerprintTrainer initialization."""
        assert trainer is not None
        assert trainer.latent_dim > 0
        assert trainer.learning_rate > 0
    
    def test_build_model(self, trainer):
        """Test model building."""
        trainer.build_model(input_dim=128)
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
    
    def test_train(self, trainer, sample_features):
        """Test training."""
        trainer.build_model(input_dim=128)
        
        # Split data
        train_data = sample_features[:150]
        val_data = sample_features[150:]
        
        history = trainer.train(train_data, val_data)
        
        assert history is not None
        assert 'train_losses' in history
        assert len(history['train_losses']) > 0
    
    def test_generate_fingerprints(self, trainer, sample_features):
        """Test fingerprint generation."""
        trainer.build_model(input_dim=128)
        trainer.train(sample_features, sample_features[:100])
        
        fingerprints = trainer.generate_fingerprints(sample_features)
        
        assert fingerprints is not None
        assert fingerprints.shape == (sample_features.shape[0], trainer.latent_dim)
    
    def test_save_load_checkpoint(self, trainer, sample_features, tmp_path):
        """Test checkpoint save and load."""
        trainer.build_model(input_dim=128)
        trainer.train(sample_features, sample_features[:100])
        
        # Save checkpoint
        save_path = tmp_path / 'model.pt'
        trainer.save_checkpoint(str(save_path))
        
        # Load checkpoint
        new_trainer = FingerprintTrainer(self.config)
        new_trainer.build_model(input_dim=128)
        new_trainer.load_checkpoint(str(save_path))
        
        assert new_trainer.model is not None


class TestAnomalyDetector:
    """Test anomaly detection module."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def detector(self, config):
        """Create AnomalyDetector instance."""
        return AnomalyDetector(config)
    
    @pytest.fixture
    def sample_fingerprints(self):
        """Create sample fingerprints."""
        # Normal data
        normal = np.random.randn(100, 32) * 0.1
        # Anomalous data
        anomalous = np.random.randn(20, 32) * 1.0 + 2
        return np.vstack([normal, anomalous])
    
    def test_initialization(self, detector):
        """Test AnomalyDetector initialization."""
        assert detector is not None
        assert detector.threshold_percentile == 95
        assert detector.contamination == 0.1
    
    def test_fit(self, detector, sample_fingerprints):
        """Test fitting the detector."""
        detector.fit(sample_fingerprints)
        
        assert detector.is_fitted is True
        assert detector.isolation_forest is not None
        assert detector.lof is not None
        assert detector.threshold is not None
    
    def test_predict(self, detector, sample_fingerprints):
        """Test prediction."""
        detector.fit(sample_fingerprints)
        
        results = detector.predict(sample_fingerprints)
        
        assert 'scores' in results
        assert 'predictions' in results
        assert 'probabilities' in results
        assert len(results['scores']) == len(sample_fingerprints)
    
    def test_save_load(self, detector, sample_fingerprints, tmp_path):
        """Test save and load."""
        detector.fit(sample_fingerprints)
        
        # Save
        save_path = tmp_path / 'detector.pkl'
        detector.save(str(save_path))
        
        # Load
        new_detector = AnomalyDetector(self.config)
        new_detector.load(str(save_path))
        
        assert new_detector.is_fitted is True
        assert new_detector.threshold == detector.threshold


class TestSmartScheduler:
    """Test smart scheduler."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def scheduler(self, config):
        """Create SmartScheduler instance."""
        return SmartScheduler(config)
    
    def test_initialization(self, scheduler):
        """Test SmartScheduler initialization."""
        assert scheduler is not None
        assert len(scheduler.resource_weights) > 0
    
    def test_set_cluster_state(self, scheduler):
        """Test setting cluster state."""
        capacities = {
            'node-1': {'gpu': 4, 'memory': 32, 'cpu': 16},
            'node-2': {'gpu': 2, 'memory': 16, 'cpu': 8}
        }
        utilization = {
            'node-1': {'gpu': 0.5, 'memory': 0.4, 'cpu': 0.3},
            'node-2': {'gpu': 0.2, 'memory': 0.1, 'cpu': 0.1}
        }
        
        scheduler.set_cluster_state(capacities, utilization)
        
        assert len(scheduler.node_capacities) == 2
        assert len(scheduler.node_utilizations) == 2
    
    def test_predict_resource_demand(self, scheduler):
        """Test resource demand prediction."""
        fingerprint = np.random.randn(32)
        demand = scheduler.predict_resource_demand(fingerprint)
        
        assert 'gpu' in demand
        assert 'memory' in demand
        assert 'cpu' in demand
        assert 0 <= demand['gpu'] <= 1
        assert 0 <= demand['memory'] <= 1
    
    def test_find_best_node(self, scheduler):
        """Test finding best node."""
        capacities = {
            'node-1': {'gpu': 4, 'memory': 32, 'cpu': 16},
            'node-2': {'gpu': 2, 'memory': 16, 'cpu': 8}
        }
        utilization = {
            'node-1': {'gpu': 0.5, 'memory': 0.4, 'cpu': 0.3},
            'node-2': {'gpu': 0.2, 'memory': 0.1, 'cpu': 0.1}
        }
        
        scheduler.set_cluster_state(capacities, utilization)
        
        fingerprint = np.random.randn(32)
        recommendation = scheduler.find_best_node(fingerprint, 'job-123')
        
        assert isinstance(recommendation, SchedulingRecommendation)
        assert recommendation.job_id == 'job-123'
        assert recommendation.recommended_node in ['node-1', 'node-2']


class TestCostOptimizer:
    """Test cost optimizer."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config()
    
    @pytest.fixture
    def optimizer(self, config):
        """Create CostOptimizer instance."""
        return CostOptimizer(config)
    
    def test_initialization(self, optimizer):
        """Test CostOptimizer initialization."""
        assert optimizer is not None
        assert optimizer.hourly_rate > 0
        assert optimizer.idle_threshold >= 0
    
    def test_detect_idle_gpus(self, optimizer):
        """Test idle GPU detection."""
        metrics = {
            'gpu_metrics': {
                'node-1': {
                    0: {'utilization': 0.05, 'memory_utilization': 0.2},
                    1: {'utilization': 0.85, 'memory_utilization': 0.6}
                },
                'node-2': {
                    0: {'utilization': 0.02, 'memory_utilization': 0.8},
                    1: {'utilization': 0.03, 'memory_utilization': 0.1}
                }
            }
        }
        
        idle_gpus = optimizer.detect_idle_gpus(metrics)
        
        assert len(idle_gpus) > 0
        for gpu in idle_gpus:
            assert isinstance(gpu, IdleGPUInfo)
            assert gpu.compute_utilization < optimizer.idle_threshold
    
    def test_analyze_cost_optimization(self, optimizer):
        """Test cost optimization analysis."""
        metrics = {
            'gpu_metrics': {
                'node-1': {
                    0: {'utilization': 0.05, 'memory_utilization': 0.2},
                    1: {'utilization': 0.85, 'memory_utilization': 0.6}
                }
            }
        }
        
        result = optimizer.analyze_cost_optimization(metrics)
        
        assert result.total_idle_gpus >= 0
        assert result.total_cost_wasted >= 0
        assert len(result.recommendations) > 0
    
    def test_generate_savings_report(self, optimizer):
        """Test savings report generation."""
        metrics = {
            'gpu_metrics': {
                'node-1': {
                    0: {'utilization': 0.05, 'memory_utilization': 0.2}
                }
            }
        }
        
        result = optimizer.analyze_cost_optimization(metrics)
        report = optimizer.generate_savings_report(result)
        
        assert 'summary' in report
        assert 'idle_gpus' in report
        assert 'recommendations' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])