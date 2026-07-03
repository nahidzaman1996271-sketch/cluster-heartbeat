"""
Dependencies for API routes.
Provides dependency injection for services and configuration.
"""

from typing import Dict, Any, Optional
import functools
import time
from fastapi import Request, HTTPException, Depends
from ..core.service import ClusterHeartbeatService
from ..config import load_config, get_config as get_config_value
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Singleton service instance
_service_instance: Optional[ClusterHeartbeatService] = None
_service_lock = None  # Will be initialized if needed


def get_service() -> ClusterHeartbeatService:
    """
    Get or create the Cluster Heartbeat service instance.
    
    Returns:
        ClusterHeartbeatService instance
    """
    global _service_instance
    
    if _service_instance is None:
        config = get_config_value()
        _service_instance = ClusterHeartbeatService(config)
        _service_instance.start()
        logger.info("ClusterHeartbeatService instance created and started")
    
    return _service_instance


def get_config() -> Dict[str, Any]:
    """
    Get configuration.
    
    Returns:
        Configuration dictionary
    """
    return load_config()


# ============================================
# Rate Limiting Dependencies
# ============================================

class RateLimiter:
    """
    Simple rate limiter for API endpoints.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if client is allowed to make a request.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if allowed, False otherwise
        """
        now = time.time()
        window = 60  # 1 minute
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < window
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """
    Get rate limiter instance.
    
    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        config = get_config()
        rate_config = config.get('api', {}).get('rate_limit', {})
        requests_per_minute = rate_config.get('requests_per_minute', 60)
        _rate_limiter = RateLimiter(requests_per_minute)
    
    return _rate_limiter


async def rate_limit(request: Request) -> None:
    """
    Rate limiting dependency.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    config = get_config()
    rate_config = config.get('api', {}).get('rate_limit', {})
    
    if not rate_config.get('enabled', True):
        return
    
    # Get client identifier
    client_id = request.client.host if request.client else "unknown"
    
    # Check rate limit
    limiter = get_rate_limiter()
    if not limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )


# ============================================
# Authentication Dependencies
# ============================================

async def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """
    Get current authenticated user.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User information or None
    """
    config = get_config()
    auth_config = config.get('api', {}).get('security', {})
    
    if not auth_config.get('enabled', False):
        return {"user": "anonymous", "role": "viewer"}
    
    # Get token from header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    
    try:
        # Extract token
        token = auth_header.split(" ")[1] if " " in auth_header else auth_header
        
        # Validate token (simplified)
        # In production, use JWT validation
        # For now, check if token matches expected
        expected_token = auth_config.get('secret_key', '')
        if token == expected_token:
            return {"user": "authenticated", "role": "admin"}
        
        return None
    
    except Exception as e:
        logger.warning(f"Authentication error: {e}")
        return None


async def require_auth(request: Request) -> Dict[str, Any]:
    """
    Require authentication.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User information
        
    Raises:
        HTTPException: If not authenticated
    """
    user = await get_current_user(request)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    return user


async def require_admin(request: Request) -> Dict[str, Any]:
    """
    Require admin role.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User information
        
    Raises:
        HTTPException: If not admin
    """
    user = await get_current_user(request)
    if user is None or user.get('role') != 'admin':
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    return user


# ============================================
# Validation Dependencies
# ============================================

async def validate_metrics_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate metrics data.
    
    Args:
        data: Metrics data to validate
        
    Returns:
        Validated data
        
    Raises:
        HTTPException: If validation fails
    """
    required_fields = ['timestamp', 'metrics']
    
    for field in required_fields:
        if field not in data:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field: {field}"
            )
    
    # Validate metrics
    metrics = data.get('metrics', {})
    if not isinstance(metrics, dict):
        raise HTTPException(
            status_code=400,
            detail="Metrics must be a dictionary"
        )
    
    # Check for required metric fields
    required_metrics = ['gpu_utilization', 'memory_utilization']
    for metric in required_metrics:
        if metric not in metrics:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required metric: {metric}"
            )
    
    return data


# ============================================
# Logging Dependencies
# ============================================

async def log_request(request: Request, call_next):
    """
    Log all requests.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware
        
    Returns:
        Response
    """
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request
    elapsed = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Time: {elapsed:.3f}s"
    )
    
    return response


# ============================================
# Pagination Dependencies
# ============================================

def get_pagination_params(page: int = 1, limit: int = 100) -> Dict[str, int]:
    """
    Get pagination parameters.
    
    Args:
        page: Page number
        limit: Items per page
        
    Returns:
        Pagination parameters
    """
    if page < 1:
        page = 1
    if limit < 1:
        limit = 10
    if limit > 1000:
        limit = 1000
    
    return {
        "page": page,
        "limit": limit,
        "skip": (page - 1) * limit
    }


def create_pagination_response(data: List[Any], total: int, page: int, limit: int) -> Dict[str, Any]:
    """
    Create pagination response.
    
    Args:
        data: List of items
        total: Total items
        page: Current page
        limit: Items per page
        
    Returns:
        Paginated response
    """
    total_pages = (total + limit - 1) // limit
    
    return {
        "data": data,
        "pagination": {
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1
        }
    }


# ============================================
# Sort Dependencies
# ============================================

def get_sort_params(sort_by: Optional[str] = None, sort_order: str = "asc") -> Dict[str, Any]:
    """
    Get sort parameters.
    
    Args:
        sort_by: Field to sort by
        sort_order: Sort order (asc/desc)
        
    Returns:
        Sort parameters
    """
    allowed_fields = [
        'timestamp', 'health_score', 'gpu_utilization', 
        'memory_utilization', 'node_id', 'job_id'
    ]
    
    if sort_by and sort_by not in allowed_fields:
        sort_by = 'timestamp'
    
    if sort_order.lower() not in ['asc', 'desc']:
        sort_order = 'asc'
    
    return {
        "sort_by": sort_by or "timestamp",
        "sort_order": sort_order.lower()
    }


# ============================================
# API Version Dependencies
# ============================================

def get_api_version(version: str = "v1") -> str:
    """
    Get API version.
    
    Args:
        version: Requested version
        
    Returns:
        API version
    """
    supported_versions = ["v1"]
    
    if version not in supported_versions:
        version = "v1"
    
    return version


# ============================================
# Export all dependencies
# ============================================

__all__ = [
    'get_service',
    'get_config',
    'rate_limit',
    'get_rate_limiter',
    'get_current_user',
    'require_auth',
    'require_admin',
    'validate_metrics_data',
    'log_request',
    'get_pagination_params',
    'create_pagination_response',
    'get_sort_params',
    'get_api_version'
]