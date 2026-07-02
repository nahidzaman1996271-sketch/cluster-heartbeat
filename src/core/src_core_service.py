"""
Service layer that integrates the pipeline with API and background processing.
"""

import asyncio
import threading
import queue
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import time

from .pipeline import ClusterHeartbeatPipeline
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ClusterHeartbeatService:
    """
    Main service class that orchestrates the entire system.
    Handles background processing, caching, and API integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pipeline = ClusterHeartbeatPipeline(config)
        self.pipeline.initialize(load_models=True)
        
        # Background processing
        self.is_running = False
        self.bg_thread = None
        self.processing_queue = queue.Queue()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 100
        
        # State
        self.last_processed = None
        self.health_status = "initializing"
        self.start_time = datetime.now()
        self.stats = {
            'total_jobs': 0,
            'processed_jobs': 0,
            'failed_jobs': 0,
            'avg_processing_time': 0,
            'total_processing_time': 0
        }
        
        logger.info("Cluster Heartbeat Service initialized")
    
    def start(self):
        """Start background processing."""
        if self.is_running:
            logger.warning("Service already running")
            return
        
        self.is_running = True
        self.bg_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.bg_thread.start()
        self.health_status = "running"
        logger.info("Background processing started")
    
    def stop(self):
        """Stop background processing."""
        self.is_running = False
        if self.bg_thread:
            self.bg_thread.join(timeout=5)
        self.health_status = "stopped"
        logger.info("Background processing stopped")
    
    def process_metrics(self, metrics_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Process incoming metrics synchronously.
        
        Args:
            metrics_data: Incoming metrics (single record or list)
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            if isinstance(metrics_data, dict):
                df = pd.DataFrame([metrics_data])
            else:
                df = pd.DataFrame(metrics_data)
            
            # Process through pipeline
            results = self.pipeline.process_batch(df)
            
            # Update cache
            cache_key = f"results_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.cache[cache_key] = {
                'results': results,
                'timestamp': datetime.now()
            }
            self.cache['last_results'] = results
            self.cache['last_results_time'] = datetime.now()
            
            # Clean old cache
            self._clean_cache()
            
            # Update stats
            self.stats['processed_jobs'] += 1
            self.stats['total_processing_time'] += time.time() - start_time
            self.stats['avg_processing_time'] = self.stats['total_processing_time'] / self.stats['processed_jobs']
            
            self.last_processed = datetime.now()
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing metrics: {e}", exc_info=True)
            self.stats['failed_jobs'] += 1
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def process_async(self, metrics_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """
        Queue metrics for async processing.
        
        Args:
            metrics_data: Incoming metrics
            
        Returns:
            Job ID for tracking
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.stats['total_jobs']}"
        
        self.processing_queue.put({
            'job_id': job_id,
            'data': metrics_data,
            'timestamp': datetime.now(),
            'status': 'queued'
        })
        
        self.stats['total_jobs'] += 1
        
        logger.info(f"Queued job: {job_id}")
        return job_id
    
    def _background_worker(self):
        """Background worker for async processing."""
        while self.is_running:
            try:
                # Get next job with timeout
                job = self.processing_queue.get(timeout=1)
                
                # Process
                logger.info(f"Processing async job: {job['job_id']}")
                job['status'] = 'processing'
                results = self.process_metrics(job['data'])
                
                # Store results
                self.cache[job['job_id']] = {
                    'results': results,
                    'timestamp': datetime.now(),
                    'status': 'completed'
                }
                
                # Clean old cache
                self._clean_cache()
                
                logger.info(f"Completed async job: {job['job_id']}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Background worker error: {e}", exc_info=True)
                # Store error in cache
                job_id = job.get('job_id', 'unknown')
                self.cache[job_id] = {
                    'error': str(e),
                    'timestamp': datetime.now(),
                    'status': 'failed'
                }
    
    def _clean_cache(self):
        """Clean old cache entries."""
        now = datetime.now()
        to_delete = []
        
        for key, value in self.cache.items():
            # Skip last_results
            if key == 'last_results' or key == 'last_results_time':
                continue
            
            if isinstance(value, dict) and 'timestamp' in value:
                age = (now - value['timestamp']).seconds
                if age > self.cache_ttl:
                    to_delete.append(key)
        
        for key in to_delete:
            del self.cache[key]
        
        # Limit cache size
        if len(self.cache) > self.max_cache_size:
            # Sort by timestamp and remove oldest
            sorted_items = sorted(
                [(k, v['timestamp']) for k, v in self.cache.items() 
                 if isinstance(v, dict) and 'timestamp' in v],
                key=lambda x: x[1]
            )
            for key, _ in sorted_items[:len(self.cache) - self.max_cache_size]:
                if key not in ['last_results', 'last_results_time']:
                    del self.cache[key]
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status
        """
        if job_id in self.cache:
            return {
                'job_id': job_id,
                'status': self.cache[job_id].get('status', 'unknown'),
                'timestamp': self.cache[job_id].get('timestamp', datetime.now()).isoformat(),
                'has_results': 'results' in self.cache[job_id]
            }
        else:
            # Check if still in queue
            for q_item in list(self.processing_queue.queue):
                if q_item.get('job_id') == job_id:
                    return {
                        'job_id': job_id,
                        'status': 'queued',
                        'timestamp': q_item.get('timestamp', datetime.now()).isoformat()
                    }
            
            return {
                'job_id': job_id,
                'status': 'not_found',
                'error': 'Job not found'
            }
    
    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """
        Get results of a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job results
        """
        if job_id in self.cache and 'results' in self.cache[job_id]:
            return self.cache[job_id]['results']
        else:
            return {
                'error': 'Results not found',
                'job_id': job_id,
                'status': self.get_job_status(job_id).get('status', 'unknown')
            }
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get current cluster status.
        
        Returns:
            Cluster status dictionary
        """
        if 'last_results' not in self.cache:
            return {
                'status': 'unknown',
                'message': 'No data processed yet',
                'timestamp': datetime.now().isoformat()
            }
        
        results = self.cache['last_results']
        summary = results.get('summary', {})
        
        return {
            'status': summary.get('health_status', 'unknown'),
            'timestamp': results.get('timestamp'),
            'nodes': summary.get('total_nodes', 0),
            'average_health': summary.get('average_health', 0),
            'anomalies': summary.get('anomaly_count', 0),
            'idle_gpus': summary.get('idle_gpus', 0),
            'potential_savings': summary.get('cost_savings', 0),
            'total_cost_wasted': summary.get('total_cost_wasted', 0),
            'service_uptime': (datetime.now() - self.start_time).seconds,
            'service_stats': {
                'total_jobs': self.stats['total_jobs'],
                'processed_jobs': self.stats['processed_jobs'],
                'failed_jobs': self.stats['failed_jobs'],
                'avg_processing_time': self.stats['avg_processing_time']
            }
        }
    
    def get_health_score(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health score for cluster or specific node.
        
        Args:
            node_id: Optional node identifier
            
        Returns:
            Health score
        """
        if 'last_results' not in self.cache:
            return {'error': 'No data available'}
        
        results = self.cache['last_results']
        health_scores = results.get('health_scores', {})
        
        if node_id:
            score = health_scores.get('node_scores', {}).get(node_id, 0)
            return {
                'node_id': node_id,
                'health_score': score,
                'status': self._get_health_status(score)
            }
        else:
            return {
                'average_health': health_scores.get('average_health', 0),
                'status': health_scores.get('status', 'unknown'),
                'node_count': len(health_scores.get('node_scores', {}))
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
    
    def get_predictions(self) -> Dict[str, Any]:
        """
        Get predictions (failure risk, etc.).
        
        Returns:
            Prediction results
        """
        if 'last_results' not in self.cache:
            return {'error': 'No data available'}
        
        results = self.cache['last_results']
        anomaly_results = results.get('anomaly_results', {})
        health_scores = results.get('health_scores', {})
        
        predictions = []
        scores = anomaly_results.get('scores', [])
        probabilities = anomaly_results.get('probabilities', [])
        preds = anomaly_results.get('predictions', [])
        
        for i, score in enumerate(scores):
            if i < len(probabilities):
                predictions.append({
                    'index': i,
                    'anomaly_score': score,
                    'probability': probabilities[i] if i < len(probabilities) else 0,
                    'is_anomaly': bool(preds[i]) if i < len(preds) else False,
                    'health': health_scores.get('health_scores', [100])[i] if i < len(health_scores.get('health_scores', [])) else 100
                })
        
        return {
            'predictions': predictions,
            'total_anomalies': int(sum(preds)) if preds else 0,
            'threshold': anomaly_results.get('threshold', 0),
            'timestamp': results.get('timestamp')
        }
    
    def get_scheduling_recommendations(self) -> Dict[str, Any]:
        """
        Get scheduling recommendations.
        
        Returns:
            Scheduling recommendations
        """
        if 'last_results' not in self.cache:
            return {'error': 'No data available'}
        
        results = self.cache['last_results']
        return {
            'recommendations': results.get('scheduling', []),
            'total': len(results.get('scheduling', [])),
            'timestamp': results.get('timestamp')
        }
    
    def get_cost_savings(self) -> Dict[str, Any]:
        """
        Get cost saving recommendations.
        
        Returns:
            Cost saving recommendations
        """
        if 'last_results' not in self.cache:
            return {'error': 'No data available'}
        
        results = self.cache['last_results']
        cost_results = results.get('cost_optimization', {})
        
        return {
            'savings': cost_results,
            'timestamp': results.get('timestamp')
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data.
        
        Returns:
            Dashboard-ready JSON
        """
        if 'last_results' not in self.cache:
            return {'error': 'No data available'}
        
        return {
            'cluster_summary': self.get_cluster_status(),
            'health_scores': self.get_health_score(),
            'predictions': self.get_predictions(),
            'scheduling': self.get_scheduling_recommendations(),
            'cost_savings': self.get_cost_savings(),
            'timeseries': self._get_timeseries_data(),
            'service_info': {
                'uptime': (datetime.now() - self.start_time).seconds,
                'version': self.config.get('project', {}).get('version', '1.0.0'),
                'environment': self.config.get('project', {}).get('environment', 'development')
            }
        }
    
    def _get_timeseries_data(self) -> Dict[str, Any]:
        """Get time series data for dashboard."""
        if 'last_results' not in self.cache:
            return {'timestamps': [], 'metrics': {}}
        
        # Return cached metrics
        return self.pipeline.get_dashboard_data().get('timeseries_metrics', {
            'timestamps': [],
            'metrics': {}
        })
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Service statistics
        """
        return {
            'uptime_seconds': (datetime.now() - self.start_time).seconds,
            'total_jobs': self.stats['total_jobs'],
            'processed_jobs': self.stats['processed_jobs'],
            'failed_jobs': self.stats['failed_jobs'],
            'avg_processing_time': self.stats['avg_processing_time'],
            'cache_size': len(self.cache),
            'queue_size': self.processing_queue.qsize(),
            'is_running': self.is_running,
            'health_status': self.health_status,
            'last_processed': self.last_processed.isoformat() if self.last_processed else None
        }
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def shutdown(self):
        """Shutdown the service."""
        self.stop()
        self.pipeline.shutdown()
        self.health_status = "shutdown"
        logger.info("Service shutdown complete")


# Convenience function
def create_service(config: Dict[str, Any]) -> ClusterHeartbeatService:
    """
    Quick function to create service.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ClusterHeartbeatService instance
    """
    return ClusterHeartbeatService(config)