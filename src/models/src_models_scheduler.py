"""
Cost optimization module for Cluster Heartbeat.
Identifies idle GPU spend and provides optimization recommendations.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IdleGPUInfo:
    """Information about idle GPU resources."""
    node_id: str
    gpu_id: int
    idle_duration: float  # seconds
    memory_utilization: float
    compute_utilization: float
    cost_wasted: float  # USD
    recommendation: str
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'gpu_id': self.gpu_id,
            'idle_duration': self.idle_duration,
            'memory_utilization': self.memory_utilization,
            'compute_utilization': self.compute_utilization,
            'cost_wasted': self.cost_wasted,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp
        }


@dataclass
class CostSavingRecommendation:
    """Container for cost saving recommendations."""
    total_idle_gpus: int
    total_cost_wasted: float
    idle_gpus: List[IdleGPUInfo]
    potential_savings: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_idle_gpus': self.total_idle_gpus,
            'total_cost_wasted': self.total_cost_wasted,
            'idle_gpus': [gpu.to_dict() for gpu in self.idle_gpus],
            'potential_savings': self.potential_savings,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class CostOptimizer:
    """
    Identifies and optimizes GPU cost leaks.
    Detects idle GPUs and provides optimization recommendations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cost optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        cost_config = config['model']['cost']
        
        # Parameters
        self.hourly_rate = cost_config.get('gpu_hourly_rate', 4.0)
        self.idle_threshold = cost_config.get('idle_threshold', 0.1)
        self.detection_window = cost_config.get('detection_window', 600)
        self.memory_idle_threshold = cost_config.get('memory_idle_threshold', 0.5)
        
        # Alerting
        self.alert_config = cost_config.get('alerts', {})
        self.idle_gpu_threshold = self.alert_config.get('idle_gpu_threshold', 2)
        self.cost_waste_threshold = self.alert_config.get('cost_waste_threshold', 100.0)
        
        # State
        self.history = {}
        self.alert_history = []
        
        logger.info("CostOptimizer initialized")
    
    def detect_idle_gpus(self, metrics: Dict[str, Any]) -> List[IdleGPUInfo]:
        """
        Detect idle GPUs based on telemetry data.
        
        Args:
            metrics: GPU telemetry data
            
        Returns:
            List of idle GPU information
        """
        idle_gpus = []
        current_time = datetime.now().timestamp()
        
        # Extract GPU metrics
        gpu_metrics = metrics.get('gpu_metrics', {})
        
        for node_id, node_metrics in gpu_metrics.items():
            for gpu_id, gpu_data in node_metrics.items():
                # Get metrics
                compute_util = gpu_data.get('utilization', 1.0)
                memory_util = gpu_data.get('memory_utilization', 0.5)
                
                # Check if GPU is idle (compute utilization below threshold)
                if compute_util < self.idle_threshold:
                    # Calculate idle duration
                    gpu_key = f"{node_id}_{gpu_id}"
                    
                    # Get previous state
                    if gpu_key in self.history:
                        idle_start = self.history[gpu_key].get('idle_start', current_time)
                        idle_duration = current_time - idle_start
                    else:
                        idle_duration = self.detection_window * (1 - compute_util / 0.1)
                        idle_start = current_time - idle_duration
                    
                    # Calculate cost wasted
                    cost_wasted = (idle_duration / 3600) * self.hourly_rate
                    
                    # Generate recommendation
                    if memory_util > self.memory_idle_threshold:
                        recommendation = (
                            f"Memory allocated ({memory_util:.1%}) but no compute "
                            f"({compute_util:.1%}) - consider releasing or resizing"
                        )
                    else:
                        recommendation = (
                            f"GPU idle ({compute_util:.1%} utilization) - "
                            f"consider scaling down or consolidating workloads"
                        )
                    
                    idle_gpus.append(IdleGPUInfo(
                        node_id=str(node_id),
                        gpu_id=int(gpu_id),
                        idle_duration=idle_duration,
                        memory_utilization=float(memory_util),
                        compute_utilization=float(compute_util),
                        cost_wasted=float(cost_wasted),
                        recommendation=recommendation,
                        timestamp=current_time
                    ))
                    
                    # Update history
                    if gpu_key not in self.history:
                        self.history[gpu_key] = {}
                    self.history[gpu_key]['idle_start'] = idle_start
                else:
                    # GPU is active
                    gpu_key = f"{node_id}_{gpu_id}"
                    if gpu_key in self.history:
                        # Clear idle tracking
                        self.history[gpu_key].pop('idle_start', None)
        
        return idle_gpus
    
    def analyze_cost_optimization(self, cluster_metrics: Dict[str, Any]) -> CostSavingRecommendation:
        """
        Analyze cluster metrics for cost optimization opportunities.
        
        Args:
            cluster_metrics: Complete cluster telemetry
            
        Returns:
            Cost saving recommendation
        """
        # Detect idle GPUs
        idle_gpus = self.detect_idle_gpus(cluster_metrics)
        
        # Calculate totals
        total_idle = len(idle_gpus)
        total_cost = sum(gpu.cost_wasted for gpu in idle_gpus)
        
        # Generate recommendations
        recommendations = []
        
        # 1. Idle GPU recommendations
        if total_idle > 0:
            # Group by node
            node_gpus = {}
            for gpu in idle_gpus:
                if gpu.node_id not in node_gpus:
                    node_gpus[gpu.node_id] = []
                node_gpus[gpu.node_id].append(gpu)
            
            for node_id, gpus in node_gpus.items():
                total_node_gpus = len(cluster_metrics.get('gpu_metrics', {}).get(node_id, {}))
                
                if len(gpus) == total_node_gpus:
                    recommendations.append(
                        f"All {total_node_gpus} GPUs on node {node_id} are idle - "
                        f"consider shutting down node (savings: ${sum(g.cost_wasted for g in gpus):.2f})"
                    )
                elif len(gpus) > self.idle_gpu_threshold:
                    recommendations.append(
                        f"Node {node_id} has {len(gpus)} idle GPUs - "
                        f"consider consolidating workloads (savings: ${sum(g.cost_wasted for g in gpus):.2f})"
                    )
        
        # 2. Cost threshold recommendation
        if total_cost > self.cost_waste_threshold:
            recommendations.append(
                f"High idle GPU cost detected: ${total_cost:.2f} wasted - "
                f"review workload scheduling immediately"
            )
        
        # 3. Memory-specific recommendations
        memory_idle = [gpu for gpu in idle_gpus if gpu.memory_utilization > self.memory_idle_threshold]
        if memory_idle:
            recommendations.append(
                f"{len(memory_idle)} GPUs have memory allocated but no compute - "
                f"potential savings: ${sum(g.cost_wasted for g in memory_idle):.2f}"
            )
        
        # 4. Best practice recommendations
        if total_idle > 0:
            recommendations.append(
                f"Consider implementing automated scaling to reduce idle GPU time"
            )
        else:
            recommendations.append("No idle GPUs detected - cluster utilization is optimal")
        
        # Calculate potential savings (conservative estimate)
        potential_savings = total_cost * 0.8  # 80% of wasted cost
        
        # Create recommendation
        return CostSavingRecommendation(
            total_idle_gpus=total_idle,
            total_cost_wasted=total_cost,
            idle_gpus=idle_gpus,
            potential_savings=potential_savings,
            recommendations=recommendations
        )
    
    def generate_savings_report(self, recommendations: CostSavingRecommendation) -> Dict[str, Any]:
        """
        Generate a formatted savings report.
        
        Args:
            recommendations: Cost saving recommendations
            
        Returns:
            Formatted report
        """
        return {
            'summary': {
                'total_idle_gpus': recommendations.total_idle_gpus,
                'total_cost_wasted': recommendations.total_cost_wasted,
                'potential_savings': recommendations.potential_savings,
                'timestamp': recommendations.timestamp.isoformat()
            },
            'idle_gpus': [
                {
                    'node_id': gpu.node_id,
                    'gpu_id': gpu.gpu_id,
                    'idle_duration': gpu.idle_duration,
                    'memory_utilization': gpu.memory_utilization,
                    'compute_utilization': gpu.compute_utilization,
                    'cost_wasted': gpu.cost_wasted,
                    'recommendation': gpu.recommendation
                }
                for gpu in recommendations.idle_gpus
            ],
            'recommendations': recommendations.recommendations,
            'metrics': {
                'hourly_rate': self.hourly_rate,
                'idle_threshold': self.idle_threshold,
                'detection_window': self.detection_window,
                'memory_idle_threshold': self.memory_idle_threshold
            }
        }
    
    def get_cost_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        Get cost trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Cost trends dictionary
        """
        # In production, this would query a database
        # For now, generate sample trends
        timestamps = [datetime.now() - timedelta(days=i) for i in range(days)]
        
        return {
            'timestamps': [ts.isoformat() for ts in reversed(timestamps)],
            'daily_cost': [100 + 50 * np.sin(i/2) + 20 * np.random.randn() for i in range(days)],
            'idle_gpus': [2 + 3 * np.sin(i/3) + np.random.randint(0, 2) for i in range(days)],
            'total_wasted': [100 + 50 * np.sin(i/2) + 20 * np.random.randn() for i in range(days)]
        }
    
    def get_cost_metrics(self, cluster_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get cost metrics for the cluster.
        
        Args:
            cluster_metrics: Cluster telemetry
            
        Returns:
            Cost metrics
        """
        # Calculate total GPU cost
        gpu_metrics = cluster_metrics.get('gpu_metrics', {})
        total_gpus = sum(len(node_metrics) for node_metrics in gpu_metrics.values())
        
        # Calculate effective utilization
        total_utilization = 0
        for node_metrics in gpu_metrics.values():
            for gpu_data in node_metrics.values():
                total_utilization += gpu_data.get('utilization', 0.5)
        
        avg_utilization = total_utilization / total_gpus if total_gpus > 0 else 0
        
        return {
            'total_gpus': total_gpus,
            'average_utilization': avg_utilization,
            'estimated_daily_cost': total_gpus * self.hourly_rate * 24,
            'estimated_wasted_cost': total_gpus * self.hourly_rate * 24 * (1 - avg_utilization),
            'idle_gpus_detected': len(self.detect_idle_gpus(cluster_metrics))
        }


# Convenience function
def create_cost_optimizer(config: Dict[str, Any]) -> CostOptimizer:
    """
    Quick function to create cost optimizer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CostOptimizer instance
    """
    return CostOptimizer(config)