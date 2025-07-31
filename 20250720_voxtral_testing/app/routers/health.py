"""Health check endpoints."""

from datetime import datetime
from fastapi import APIRouter, HTTPException, status

from ..core.config import config
from ..core.logging import get_logger
from ..models.audio import HealthResponse
from ..services.voxtral_service import voxtral_service

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health of the service and underlying model."""
    try:
        # Check if Voxtral backend is healthy
        is_healthy = await voxtral_service.health_check()
        
        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Voxtral backend is not available"
            )
        
        # Get model name
        model_name = await voxtral_service.get_model_name()
        
        return HealthResponse(
            status="healthy",
            model=model_name,
            timestamp=datetime.utcnow(),
            version=config.app_version,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service health check failed: {str(e)}"
        )
