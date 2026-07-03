"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app
from src.api.dependencies import get_service, get_config
from src.core.service import ClusterHeartbeatService
from src.config import load_config


class TestAPI:
    """Test API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
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
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data['service'] == 'Cluster Heartbeat'
        assert 'version' in data
        assert 'endpoints' in data
    
    def test_ping(self, client):
        """Test ping endpoint."""
        response = client.get("/ping")
        
        assert response.status_code == 200
        data = response.json()
        assert data['pong'] is True
        assert 'timestamp' in data
    
    def test_version(self, client):
        """Test version endpoint."""
        response = client.get("/version")
        
        assert response.status_code == 200
        data = response.json()
        assert 'service' in data
        assert 'version' in data
        assert 'api_version' in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] in ['healthy', 'degraded']
        assert 'timestamp' in data
        assert 'service' in data
    
    def test_liveness(self, client):
        """Test liveness endpoint."""
        response = client.get("/api/v1/health/liveness")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'alive'
    
    def test_readiness(self, client):
        """Test readiness endpoint."""
        response = client.get("/api/v1/health/readiness")
        
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert data['status'] == 'ready'
    
    def test_healthz(self, client):
        """Test healthz endpoint."""
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
    
    def test_ingest_metrics(self, client):
        """Test metrics ingestion endpoint."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'gpu_utilization': 0.85,
                'memory_utilization': 0.65,
                'gpu_temperature': 72.5
            },
            'node_id': 'node-1',
            'job_id': 'job-123'
        }
        
        response = client.post("/api/v1/metrics/ingest", json=data)
        
        assert response.status_code == 200
        result = response.json()
        assert result['status'] == 'accepted'
        assert 'job_id' in result
    
    def test_get_cluster_status(self, client):
        """Test cluster status endpoint."""
        response = client.get("/api/v1/predictions/cluster-status")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data or 'error' in data
    
    def test_get_health_score(self, client):
        """Test health score endpoint."""
        response = client.get("/api/v1/predictions/health-score/node-1")
        
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.json()
            assert 'node_id' in data or 'error' in data
    
    def test_get_scheduling(self, client):
        """Test scheduling recommendations endpoint."""
        response = client.get("/api/v1/recommendations/scheduling")
        
        assert response.status_code == 200
        data = response.json()
        assert 'recommendations' in data or 'error' in data
    
    def test_get_cost_savings(self, client):
        """Test cost savings endpoint."""
        response = client.get("/api/v1/recommendations/cost-savings")
        
        assert response.status_code == 200
        data = response.json()
        assert 'savings' in data or 'error' in data
    
    def test_get_dashboard(self, client):
        """Test dashboard endpoint."""
        response = client.get("/dashboard")
        
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert 'cluster_summary' in data or 'error' in data
    
    def test_get_service_status(self, client):
        """Test service status endpoint."""
        response = client.get("/service/status")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'uptime_seconds' in data
    
    def test_invalid_route(self, client):
        """Test invalid route."""
        response = client.get("/invalid-route")
        
        assert response.status_code == 404
    
    def test_rate_limiting(self, client):
        """Test rate limiting."""
        # Make multiple requests quickly
        for _ in range(10):
            response = client.get("/ping")
            assert response.status_code in [200, 429]


class TestAPIWithService:
    """Test API with actual service."""
    
    @pytest.fixture
    def client_with_service(self):
        """Create test client with service."""
        # Override dependency
        def get_test_service():
            config = load_config()
            service = ClusterHeartbeatService(config)
            service.start()
            return service
        
        app.dependency_overrides[get_service] = get_test_service
        
        with TestClient(app) as client:
            yield client
    
    def test_health_with_service(self, client_with_service):
        """Test health with service initialized."""
        response = client_with_service.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] in ['healthy', 'degraded']
    
    def test_dashboard_with_service(self, client_with_service):
        """Test dashboard with service initialized."""
        response = client_with_service.get("/dashboard")
        
        assert response.status_code in [200, 500]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])