"""Heartbeat Explainer — interpret cluster health with reasoning.

Sits on top of the unified fingerprint to explain:
1. Current cluster state (summary + per-node health)
2. Healthy baseline (expected shape of metrics)
3. Deviations from healthy (when/where anomalies occur)
4. Recommended actions with transparent reasoning
5. Shape view comparison (healthy shape vs current shape)
6. Key movers (which metrics changed most)
7. Dashboard-ready visualization
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HealthyBaseline:
    """Expected healthy behavior for the cluster."""
    avg_gpu_utilization: float
    avg_memory_utilization: float
    avg_power_consumption: float
    avg_temperature: float
    acceptable_error_rate: float
    notes: str


@dataclass
class Anomaly:
    """Detected deviation from healthy baseline."""
    node_id: str
    feature: str
    current_value: float
    healthy_range: tuple
    z_score: float
    severity: str  # "low", "medium", "high", "critical"
    reason: str


@dataclass
class RecommendedAction:
    """Action to restore cluster to healthy state."""
    priority: int  # 1=urgent, 2=soon, 3=plan
    action: str
    affected_nodes: list[str]
    estimated_time_to_failure: float | None
    reason: str


@dataclass
class MetricShape:
    """Representation of cluster metric shape for visualization."""
    metric_name: str
    healthy_value: float
    current_value: float
    unit: str
    change_percent: float
    deviation: float


@dataclass
class KeyMover:
    """Top metric that moved when heartbeat drifted."""
    metric: str
    healthy_value: float
    current_value: float
    change_percent: float
    impact_score: float
    reason: str


@dataclass
class SuggestedAction:
    """Single recommended action with reasoning."""
    action: str
    reason: str
    node_affected: Optional[str] = None
    estimated_ttf_hours: Optional[float] = None


def explain_cluster(latest: dict[str, dict]) -> dict:
    """Explain current cluster state, anomalies, and actions.
    
    Args:
        latest: node_id → latest fingerprint dict (from inference pipeline)
    
    Returns:
        Explainer report with baseline, anomalies, deviations, and actions.
    """
    if not latest:
        return {"error": "no fingerprints available"}
    
    # 1. Calculate current cluster state
    cluster_state = _calculate_cluster_state(latest)
    
    # 2. Define healthy baseline
    baseline = _define_healthy_baseline(latest)
    
    # 3. Detect anomalies (deviations from healthy)
    anomalies = _detect_anomalies(latest, baseline)
    
    # 4. Generate recommended actions
    actions = _generate_recommendations(latest, anomalies)
    
    return {
        "cluster_state": cluster_state,
        "healthy_baseline": baseline.__dict__,
        "anomalies": [
            {
                "node_id": a.node_id,
                "feature": a.feature,
                "current_value": round(a.current_value, 2),
                "healthy_range": a.healthy_range,
                "z_score": round(a.z_score, 2),
                "severity": a.severity,
                "reason": a.reason,
            }
            for a in anomalies
        ],
        "recommended_actions": [
            {
                "priority": act.priority,
                "action": act.action,
                "affected_nodes": act.affected_nodes,
                "estimated_time_to_failure_hours": act.estimated_time_to_failure,
                "reason": act.reason,
            }
            for act in actions
        ],
    }


def _calculate_cluster_state(latest: dict[str, dict]) -> dict:
    """Calculate current cluster-wide metrics."""
    if not latest:
        return {}
    
    nodes_data = []
    for node_id, fp in latest.items():
        nodes_data.append({
            "node_id": node_id,
            "health": fp.get("gpu_health", 0),
            "failure_risk": fp.get("failure_risk", 0),
            "gpu_util": fp.get("demand", {}).get("gpu_utilization", 0),
            "mem_util": fp.get("demand", {}).get("memory_utilization", 0),
            "power": fp.get("demand", {}).get("power_consumption", 0),
            "anomaly": fp.get("anomaly_score", 0),
        })
    
    healthy_nodes = sum(1 for n in nodes_data if n["health"] > 70)
    at_risk_nodes = sum(1 for n in nodes_data if n["failure_risk"] > 0.3)
    
    avg_gpu = sum(n["gpu_util"] for n in nodes_data) / len(nodes_data) if nodes_data else 0
    avg_mem = sum(n["mem_util"] for n in nodes_data) / len(nodes_data) if nodes_data else 0
    avg_power = sum(n["power"] for n in nodes_data) / len(nodes_data) if nodes_data else 0
    
    return {
        "total_nodes": len(nodes_data),
        "healthy_nodes": healthy_nodes,
        "at_risk_nodes": at_risk_nodes,
        "cluster_health_pct": (healthy_nodes / len(nodes_data) * 100) if nodes_data else 0,
        "average_gpu_utilization_pct": round(avg_gpu, 1),
        "average_memory_utilization_pct": round(avg_mem, 1),
        "average_power_consumption_w": round(avg_power, 1),
        "nodes_summary": nodes_data,
    }


def _define_healthy_baseline(latest: dict[str, dict]) -> HealthyBaseline:
    """Define what 'healthy' looks like for this cluster."""
    if not latest:
        return HealthyBaseline(
            avg_gpu_utilization=50.0,
            avg_memory_utilization=45.0,
            avg_power_consumption=120.0,
            avg_temperature=55.0,
            acceptable_error_rate=0.0,
            notes="No data available; using defaults."
        )
    
    # Extract healthy nodes (high health score, low risk)
    healthy_fps = [
        fp for fp in latest.values()
        if fp.get("gpu_health", 0) > 80 and fp.get("failure_risk", 0) < 0.1
    ]
    
    if not healthy_fps:
        healthy_fps = list(latest.values())[:len(latest) // 2]
    
    if healthy_fps:
        avg_gpu = sum(fp.get("demand", {}).get("gpu_utilization", 0) for fp in healthy_fps) / len(healthy_fps)
        avg_mem = sum(fp.get("demand", {}).get("memory_utilization", 0) for fp in healthy_fps) / len(healthy_fps)
        avg_power = sum(fp.get("demand", {}).get("power_consumption", 0) for fp in healthy_fps) / len(healthy_fps)
    else:
        avg_gpu = avg_mem = 50.0
        avg_power = 120.0
    
    return HealthyBaseline(
        avg_gpu_utilization=round(avg_gpu, 1),
        avg_memory_utilization=round(avg_mem, 1),
        avg_power_consumption=round(avg_power, 1),
        avg_temperature=55.0,
        acceptable_error_rate=0.0,
        notes=f"Based on {len(healthy_fps)} healthy nodes. Expected profile for stable operations."
    )


def _detect_anomalies(latest: dict[str, dict], baseline: HealthyBaseline) -> list[Anomaly]:
    """Detect deviations from healthy baseline."""
    anomalies = []
    
    for node_id, fp in latest.items():
        demand = fp.get("demand", {})
        gpu_util = demand.get("gpu_utilization", 0)
        mem_util = demand.get("memory_utilization", 0)
        power = demand.get("power_consumption", 0)
        health = fp.get("gpu_health", 0)
        risk = fp.get("failure_risk", 0)
        anomaly_score = fp.get("anomaly_score", 0)
        features = fp.get("anomaly_features", [])
        
        # Check GPU utilization deviation
        if gpu_util > baseline.avg_gpu_utilization + 40:
            anomalies.append(Anomaly(
                node_id=node_id,
                feature="gpu_utilization",
                current_value=gpu_util,
                healthy_range=(baseline.avg_gpu_utilization - 20, baseline.avg_gpu_utilization + 20),
                z_score=(gpu_util - baseline.avg_gpu_utilization) / 15,
                severity="high" if gpu_util > 85 else "medium",
                reason=f"GPU utilization ({gpu_util:.1f}%) significantly higher than healthy baseline ({baseline.avg_gpu_utilization:.1f}%)"
            ))
        
        # Check memory utilization
        if mem_util > baseline.avg_memory_utilization + 40:
            anomalies.append(Anomaly(
                node_id=node_id,
                feature="memory_utilization",
                current_value=mem_util,
                healthy_range=(baseline.avg_memory_utilization - 20, baseline.avg_memory_utilization + 20),
                z_score=(mem_util - baseline.avg_memory_utilization) / 15,
                severity="high" if mem_util > 85 else "medium",
                reason=f"Memory utilization ({mem_util:.1f}%) exceeds healthy range"
            ))
        
        # Check health score
        if health < 50:
            anomalies.append(Anomaly(
                node_id=node_id,
                feature="gpu_health",
                current_value=health,
                healthy_range=(70, 100),
                z_score=(health - 85) / 10,
                severity="critical",
                reason=f"GPU health degraded to {health:.0f}% (critical threshold < 50%)"
            ))
        
        # Check failure risk
        if risk > 0.6:
            anomalies.append(Anomaly(
                node_id=node_id,
                feature="failure_risk",
                current_value=risk,
                healthy_range=(0, 0.2),
                z_score=risk / 0.1,
                severity="critical",
                reason=f"High failure risk {risk:.2f} (>60%); node may fail soon"
            ))
        
        # Check anomaly score
        if anomaly_score > 0.7:
            feature_list = [f.get("feature", "unknown") for f in features[:3]]
            anomalies.append(Anomaly(
                node_id=node_id,
                feature="anomaly_score",
                current_value=anomaly_score,
                healthy_range=(0, 0.3),
                z_score=anomaly_score / 0.15,
                severity="high",
                reason=f"Anomalous telemetry pattern: {', '.join(feature_list) or 'multi-feature drift'}"
            ))
    
    return sorted(anomalies, key=lambda x: (
        {"critical": 0, "high": 1, "medium": 2, "low": 3}[x.severity],
        -x.z_score
    ))


def _generate_recommendations(latest: dict[str, dict], anomalies: list[Anomaly]) -> list[RecommendedAction]:
    """Generate actionable recommendations based on anomalies."""
    actions = []
    processed_nodes = set()
    
    # Group anomalies by node
    anomalies_by_node = {}
    for anomaly in anomalies:
        if anomaly.node_id not in anomalies_by_node:
            anomalies_by_node[anomaly.node_id] = []
        anomalies_by_node[anomaly.node_id].append(anomaly)
    
    # Critical: High failure risk
    critical_nodes = [
        node_id for node_id in anomalies_by_node
        if any(a.feature == "failure_risk" and a.severity == "critical" for a in anomalies_by_node[node_id])
    ]
    if critical_nodes:
        actions.append(RecommendedAction(
            priority=1,
            action="Schedule immediate maintenance and data migration",
            affected_nodes=critical_nodes,
            estimated_time_to_failure=min(
                (latest.get(n, {}).get("ttf_hours", 12) for n in critical_nodes if n in latest),
                default=12
            ),
            reason=f"Node(s) {', '.join(critical_nodes)} have high failure risk (>60%). Estimated TTF: ~{min((latest.get(n, {}).get('ttf_hours', 12) for n in critical_nodes if n in latest), default=12):.1f}h."
        ))
        processed_nodes.update(critical_nodes)
    
    # High: GPU health degradation
    degraded_nodes = [
        node_id for node_id in anomalies_by_node
        if any(a.feature == "gpu_health" and a.severity == "critical" for a in anomalies_by_node[node_id])
    ]
    if degraded_nodes:
        actions.append(RecommendedAction(
            priority=1,
            action="Inspect GPU hardware and run diagnostics",
            affected_nodes=[n for n in degraded_nodes if n not in processed_nodes],
            estimated_time_to_failure=None,
            reason="GPU health <50% indicates hardware issues (thermal, memory, ECC errors). Run nvidia-smi -q -d ECHECK and check thermal sensors."
        ))
        processed_nodes.update(degraded_nodes)
    
    # High: Anomalous patterns
    anomalous_nodes = [
        node_id for node_id in anomalies_by_node
        if any(a.feature == "anomaly_score" and a.severity == "high" for a in anomalies_by_node[node_id])
    ]
    if anomalous_nodes:
        actions.append(RecommendedAction(
            priority=2,
            action="Investigate workload behavior and telemetry drift",
            affected_nodes=[n for n in anomalous_nodes if n not in processed_nodes],
            estimated_time_to_failure=None,
            reason="Detected multi-metric behavior shift: patterns no longer match training data (healthy fleet). Check running jobs and recent config changes."
        ))
        processed_nodes.update(anomalous_nodes)
    
    # Medium: Over-utilization
    over_util_nodes = [
        node_id for node_id in anomalies_by_node
        if any(a.feature in ["gpu_utilization", "memory_utilization"] and a.severity == "medium" for a in anomalies_by_node[node_id])
    ]
    if over_util_nodes:
        actions.append(RecommendedAction(
            priority=2,
            action="Rebalance workloads or add capacity",
            affected_nodes=[n for n in over_util_nodes if n not in processed_nodes],
            estimated_time_to_failure=None,
            reason="Sustained high utilization (>80%) reduces headroom for job scheduling and increases risk of thermal throttling."
        ))
        processed_nodes.update(over_util_nodes)
    
    return actions


def explain_node(node_id: str, fingerprint: dict) -> dict:
    """Explain a single node's health and anomalies."""
    return {
        "node_id": node_id,
        "current_state": {
            "health": fingerprint.get("gpu_health", 0),
            "failure_risk": fingerprint.get("failure_risk", 0),
            "anomaly_score": fingerprint.get("anomaly_score", 0),
            "classification": fingerprint.get("classification", {}).get("label", "unknown"),
            "demand": fingerprint.get("demand", {}),
        },
        "anomalies": [
            {
                "feature": f.get("feature", "unknown"),
                "z_score": round(f.get("z", 0), 2),
                "reason": f"Telemetry drift detected: {f.get('feature')} is {f.get('z', 0):.1f}σ away from healthy distribution"
            }
            for f in fingerprint.get("anomaly_features", [])[:5]
        ],
        "reasons": fingerprint.get("reasons", []),
    }


def _compute_shape_view(latest: dict[str, dict], baseline: HealthyBaseline) -> dict:
    """Generate shape view comparing healthy vs current metrics.
    
    Returns visualization-ready data showing metric shapes side-by-side.
    """
    shapes = []
    
    # Key metrics to compare
    metrics = [
        ("gpu_utilization", "GPU Utilization", "%", lambda d: d.get("demand", {}).get("gpu_utilization", 0)),
        ("memory_utilization", "Memory Utilization", "%", lambda d: d.get("demand", {}).get("memory_utilization", 0)),
        ("power_consumption", "Power Consumption", "W", lambda d: d.get("demand", {}).get("power_consumption", 0)),
        ("gpu_health", "GPU Health", "score", lambda d: d.get("gpu_health", 0)),
        ("failure_risk", "Failure Risk", "prob", lambda d: d.get("failure_risk", 0)),
    ]
    
    for metric_id, metric_name, unit, extractor in metrics:
        values = [extractor(fp) for fp in latest.values()]
        current_avg = sum(values) / len(values) if values else 0
        
        # Get healthy baseline value
        if metric_id == "gpu_utilization":
            healthy_val = baseline.avg_gpu_utilization
        elif metric_id == "memory_utilization":
            healthy_val = baseline.avg_memory_utilization
        elif metric_id == "power_consumption":
            healthy_val = baseline.avg_power_consumption
        elif metric_id == "gpu_health":
            healthy_val = 85.0
        elif metric_id == "failure_risk":
            healthy_val = 0.05
        else:
            healthy_val = current_avg
        
        change_pct = ((current_avg - healthy_val) / max(healthy_val, 0.1)) * 100
        deviation = abs(current_avg - healthy_val)
        
        shapes.append({
            "metric": metric_name,
            "metric_id": metric_id,
            "unit": unit,
            "healthy": round(healthy_val, 1),
            "current": round(current_avg, 1),
            "change_percent": round(change_pct, 1),
            "deviation": round(deviation, 1),
            "chart_healthy": _generate_sparkline_data(healthy_val, metric_id),
            "chart_current": _generate_sparkline_data(current_avg, metric_id),
        })
    
    return {
        "shapes": shapes,
        "visualization_type": "side-by-side-metric-comparison",
        "description": "Healthy shape (left) vs current shape (right). Visualize difference to easily spot anomalies."
    }


def _generate_sparkline_data(value: float, metric_id: str) -> list[dict]:
    """Generate synthetic sparkline data for visualization."""
    # Return a few data points that show the metric value trend
    import random
    noise = [random.uniform(-0.1, 0.1) for _ in range(5)]
    base = value
    points = [{"x": i, "y": round(max(0, base + (base * n * 0.1)), 2)} for i, n in enumerate(noise)]
    return points


def _compute_key_movers(latest: dict[str, dict], baseline: HealthyBaseline) -> dict:
    """Identify which metrics moved most when heartbeat drifted.
    
    Returns top 3-5 metrics that diverged from healthy baseline.
    """
    movers = []
    
    metrics_to_check = [
        ("gpu_utilization", "GPU Utilization", lambda d: d.get("demand", {}).get("gpu_utilization", 0), baseline.avg_gpu_utilization),
        ("memory_utilization", "Memory Utilization", lambda d: d.get("demand", {}).get("memory_utilization", 0), baseline.avg_memory_utilization),
        ("power_consumption", "Power Consumption", lambda d: d.get("demand", {}).get("power_consumption", 0), baseline.avg_power_consumption),
        ("anomaly_score", "Anomaly Score", lambda d: d.get("anomaly_score", 0), 0.2),
        ("failure_risk", "Failure Risk", lambda d: d.get("failure_risk", 0), 0.1),
        ("gpu_health", "GPU Health", lambda d: d.get("gpu_health", 0), 85),
    ]
    
    for metric_id, metric_name, extractor, healthy_baseline in metrics_to_check:
        values = [extractor(fp) for fp in latest.values()]
        current_avg = sum(values) / len(values) if values else 0
        
        change_pct = ((current_avg - healthy_baseline) / max(abs(healthy_baseline), 1)) * 100
        impact_score = abs(change_pct)
        
        if impact_score > 5:  # Only include if >5% deviation
            movers.append({
                "metric": metric_name,
                "metric_id": metric_id,
                "healthy_value": round(healthy_baseline, 2),
                "current_value": round(current_avg, 2),
                "change_percent": round(change_pct, 1),
                "impact_score": round(impact_score, 1),
                "direction": "↑" if change_pct > 0 else "↓",
                "reason": _explain_metric_change(metric_id, current_avg, healthy_baseline),
            })
    
    # Sort by impact score and return top 3
    movers.sort(key=lambda x: x["impact_score"], reverse=True)
    
    return {
        "top_movers": movers[:3],
        "total_metrics_affected": len(movers),
        "description": "Top metrics that diverged from healthy baseline when heartbeat drifted."
    }


def _explain_metric_change(metric_id: str, current: float, healthy: float) -> str:
    """Generate one-sentence reason for metric change."""
    direction = "increased" if current > healthy else "decreased"
    
    reasons = {
        "gpu_utilization": f"GPU utilization {direction} — workload intensity has changed.",
        "memory_utilization": f"Memory usage {direction} — check for memory leaks or larger batch sizes.",
        "power_consumption": f"Power draw {direction} — related to GPU and memory utilization changes.",
        "anomaly_score": f"Anomaly score {direction} — detected telemetry pattern shift from training data.",
        "failure_risk": f"Failure risk {direction} — hardware degradation or sustained thermal stress detected.",
        "gpu_health": f"GPU health {direction} — GPU health score changed due to error count or thermal issues.",
    }
    
    return reasons.get(metric_id, f"Metric {metric_id} {direction} from baseline.")


def _generate_suggested_actions_detailed(latest: dict[str, dict], anomalies: list[Anomaly], key_movers: dict) -> list[dict]:
    """Generate suggested actions with detailed reasoning, one per drift."""
    actions = []
    
    # Group by severity
    critical_anomalies = [a for a in anomalies if a.severity == "critical"]
    high_anomalies = [a for a in anomalies if a.severity == "high"]
    
    if critical_anomalies:
        critical_nodes = list(set(a.node_id for a in critical_anomalies))
        primary_issue = critical_anomalies[0]
        
        action_text = "Schedule immediate maintenance and data migration"
        if primary_issue.feature == "gpu_health":
            action_text = "Run GPU diagnostics (nvidia-smi -q -d ECHECK) and check thermal sensors"
        elif primary_issue.feature == "failure_risk":
            action_text = "Migrate workloads away from at-risk node(s) and prepare replacement"
        
        actions.append({
            "priority": 1,
            "action": action_text,
            "reason": f"Node(s) {', '.join(critical_nodes[:3])} show {primary_issue.feature} = {primary_issue.current_value:.1f}. {primary_issue.reason}",
            "affected_nodes": critical_nodes,
            "estimated_ttf_hours": min((latest.get(n, {}).get("ttf_hours", 12) for n in critical_nodes), default=12),
        })
    
    if high_anomalies:
        high_nodes = list(set(a.node_id for a in high_anomalies))
        primary_issue = high_anomalies[0]
        
        action_text = "Investigate workload behavior and resource contention"
        if primary_issue.feature == "gpu_utilization":
            action_text = "Rebalance workloads or add GPU capacity to reduce sustained utilization above 80%"
        elif primary_issue.feature == "anomaly_score":
            action_text = "Review recent job submissions and cluster configuration changes"
        
        actions.append({
            "priority": 2,
            "action": action_text,
            "reason": f"High {primary_issue.feature} detected on {len(high_nodes)} node(s). {primary_issue.reason}",
            "affected_nodes": high_nodes,
            "estimated_ttf_hours": None,
        })
    
    if not actions:
        actions.append({
            "priority": 3,
            "action": "Monitor cluster and verify normal operation",
            "reason": "No critical issues detected. Continue monitoring for any emerging trends.",
            "affected_nodes": [],
            "estimated_ttf_hours": None,
        })
    
    return actions


def explain_cluster_with_dashboard(latest: dict[str, dict]) -> dict:
    """Extended explainer with shape view, key movers, and dashboard visualization.
    
    This is the main endpoint for the Heartbeat Explainer module.
    Includes everything needed for the dashboard.
    """
    if not latest:
        return {"error": "no fingerprints available"}
    
    # 1. Calculate current cluster state
    cluster_state = _calculate_cluster_state(latest)
    
    # 2. Define healthy baseline
    baseline = _define_healthy_baseline(latest)
    
    # 3. Detect anomalies
    anomalies = _detect_anomalies(latest, baseline)
    
    # 4. Compute shape view (new)
    shape_view = _compute_shape_view(latest, baseline)
    
    # 5. Identify key movers (new)
    key_movers = _compute_key_movers(latest, baseline)
    
    # 6. Generate detailed suggested actions (new)
    suggested_actions = _generate_suggested_actions_detailed(latest, anomalies, key_movers)
    
    return {
        "cluster_state": cluster_state,
        "healthy_baseline": baseline.__dict__,
        "anomalies": [
            {
                "node_id": a.node_id,
                "feature": a.feature,
                "current_value": round(a.current_value, 2),
                "healthy_range": a.healthy_range,
                "z_score": round(a.z_score, 2),
                "severity": a.severity,
                "reason": a.reason,
            }
            for a in anomalies
        ],
        "shape_view": shape_view,
        "key_movers": key_movers,
        "suggested_actions": suggested_actions,
        "recommended_actions": [
            {
                "priority": act.priority,
                "action": act.action,
                "affected_nodes": act.affected_nodes,
                "estimated_time_to_failure_hours": act.estimated_time_to_failure,
                "reason": act.reason,
            }
            for act in _generate_recommendations(latest, anomalies)
        ],
        "dashboard_ready": True,
    }
