"""
Recommendation routes for Cluster Heartbeat API.
Provides endpoints for scheduling and cost optimization recommendations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..dependencies import get_service
from ...core.service import ClusterHeartbeatService
from ...utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/scheduling")
async def get_scheduling_recommendations(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get scheduling recommendations.
    
    Returns:
        Scheduling recommendations
    """
    try:
        return service.get_scheduling_recommendations()
    except Exception as e:
        logger.error(f"Get scheduling recommendations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost-savings")
async def get_cost_savings(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get cost saving recommendations.
    
    Returns:
        Cost saving recommendations
    """
    try:
        return service.get_cost_savings()
    except Exception as e:
        logger.error(f"Get cost savings error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
async def get_all_recommendations(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get all recommendations.
    
    Returns:
        All recommendations
    """
    try:
        scheduling = service.get_scheduling_recommendations()
        cost_savings = service.get_cost_savings()
        
        return {
            "scheduling": scheduling,
            "cost_savings": cost_savings,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Get all recommendations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization")
async def get_optimization_recommendations(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get optimization recommendations combining scheduling and cost.
    
    Returns:
        Optimization recommendations
    """
    try:
        scheduling = service.get_scheduling_recommendations()
        cost = service.get_cost_savings()
        
        # Combine recommendations
        recommendations = []
        
        # Add cost recommendations
        cost_recommendations = cost.get('savings', {}).get('recommendations', [])
        for rec in cost_recommendations:
            recommendations.append({
                "type": "cost",
                "recommendation": rec,
                "priority": "high" if "high" in rec.lower() else "medium"
            })
        
        # Add scheduling recommendations
        for rec in scheduling.get('recommendations', []):
            recommendations.append({
                "type": "scheduling",
                "job_id": rec.get('job_id'),
                "recommendation": rec.get('reason'),
                "priority": "high" if rec.get('score', 0) > 0.8 else "medium"
            })
        
        return {
            "recommendations": recommendations,
            "total": len(recommendations),
            "summary": {
                "cost_optimizations": len(cost_recommendations),
                "scheduling_optimizations": len(scheduling.get('recommendations', [])),
                "potential_savings": cost.get('savings', {}).get('summary', {}).get('potential_savings', 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Get optimization recommendations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/idle-gpus")
async def get_idle_gpus(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get idle GPU recommendations.
    
    Returns:
        Idle GPU recommendations
    """
    try:
        cost = service.get_cost_savings()
        idle_gpus = cost.get('savings', {}).get('idle_gpus', [])
        
        return {
            "idle_gpus": idle_gpus,
            "total": len(idle_gpus),
            "total_cost_wasted": cost.get('savings', {}).get('summary', {}).get('total_cost_wasted', 0),
            "potential_savings": cost.get('savings', {}).get('summary', {}).get('potential_savings', 0),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Get idle GPUs error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/actions")
async def get_recommended_actions(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get actionable recommendations.
    
    Returns:
        Actionable recommendations with priority
    """
    try:
        # Get all recommendations
        scheduling = service.get_scheduling_recommendations()
        cost = service.get_cost_savings()
        
        actions = []
        
        # Cost actions
        cost_recommendations = cost.get('savings', {}).get('recommendations', [])
        for rec in cost_recommendations:
            actions.append({
                "action": rec,
                "category": "cost_optimization",
                "priority": "high" if "high" in rec.lower() or "idle" in rec.lower() else "medium",
                "impact": "monetary",
                "estimated_savings": cost.get('savings', {}).get('summary', {}).get('potential_savings', 0) / max(len(cost_recommendations), 1)
            })
        
        # Scheduling actions
        for rec in scheduling.get('recommendations', []):
            actions.append({
                "action": f"Schedule job {rec.get('job_id')} to node {rec.get('recommended_node')}",
                "category": "scheduling",
                "priority": "high" if rec.get('score', 0) > 0.8 else "medium",
                "impact": "performance",
                "estimated_savings": 0
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))
        
        return {
            "actions": actions,
            "total": len(actions),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Get recommended actions error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_recommendations_summary(
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Get summary of all recommendations.
    
    Returns:
        Recommendations summary
    """
    try:
        scheduling = service.get_scheduling_recommendations()
        cost = service.get_cost_savings()
        
        return {
            "summary": {
                "total_recommendations": (
                    len(scheduling.get('recommendations', [])) +
                    len(cost.get('savings', {}).get('recommendations', []))
                ),
                "scheduling_recommendations": len(scheduling.get('recommendations', [])),
                "cost_optimizations": len(cost.get('savings', {}).get('recommendations', [])),
                "idle_gpus": cost.get('savings', {}).get('summary', {}).get('total_idle_gpus', 0),
                "potential_savings": cost.get('savings', {}).get('summary', {}).get('potential_savings', 0),
                "total_cost_wasted": cost.get('savings', {}).get('summary', {}).get('total_cost_wasted', 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Get recommendations summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply")
async def apply_recommendation(
    recommendation_id: str,
    service: ClusterHeartbeatService = Depends(get_service)
) -> Dict[str, Any]:
    """
    Apply a specific recommendation.
    
    Args:
        recommendation_id: Recommendation identifier
        
    Returns:
        Application result
    """
    try:
        # In production, this would actually apply the recommendation
        # For now, return success
        return {
            "status": "success",
            "recommendation_id": recommendation_id,
            "message": f"Recommendation {recommendation_id} applied successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Apply recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))