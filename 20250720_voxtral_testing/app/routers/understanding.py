"""Audio understanding endpoints."""

from fastapi import APIRouter, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..core.config import config
from ..core.logging import get_logger
from ..models.audio import (
    AudioUnderstandingRequest,
    AudioUnderstandingResponse,
)
from ..services.voxtral_service import voxtral_service

logger = get_logger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/understand", tags=["understanding"])
router.state.limiter = limiter
router.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@router.post("/", response_model=AudioUnderstandingResponse)
@limiter.limit("20/minute")  # 20 requests per minute for understanding (more compute intensive)
async def understand_audio(request: AudioUnderstandingRequest) -> AudioUnderstandingResponse:
    """
    Understand and answer questions about audio content.
    
    This endpoint accepts one or more audio files along with a question
    and returns an answer based on the audio content analysis.
    
    **Supported formats:** mp3, wav, flac, m4a, ogg
    **Maximum files per request:** 5
    **Maximum file size:** 100MB per file
    **Maximum duration:** 40 minutes per file for understanding
    """
    try:
        logger.info(f"Audio understanding request received for {len(request.audio_files)} files")
        
        # Validate number of files
        if len(request.audio_files) > 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 5 audio files allowed per understanding request"
            )
        
        # Validate audio formats
        for i, audio_file in enumerate(request.audio_files):
            if audio_file.format not in config.supported_formats:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported audio format in file {i}: {audio_file.format}. "
                           f"Supported formats: {', '.join(config.supported_formats)}"
                )
        
        # Validate question
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # Perform audio understanding
        response = await voxtral_service.understand_audio(request)
        
        logger.info(f"Audio understanding completed in {response.processing_time_ms}ms")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio understanding failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio understanding failed: {str(e)}"
        )
