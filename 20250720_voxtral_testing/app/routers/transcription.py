"""Transcription endpoints."""

from fastapi import APIRouter, HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..core.config import config
from ..core.logging import get_logger
from ..models.audio import (
    TranscriptionRequest,
    TranscriptionResponse,
    BatchTranscriptionRequest,
    BatchTranscriptionResponse,
)
from ..services.voxtral_service import voxtral_service

logger = get_logger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/transcribe", tags=["transcription"])
router.state.limiter = limiter
router.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@router.post("/", response_model=TranscriptionResponse)
@limiter.limit("50/minute")  # 50 requests per minute for transcription
async def transcribe_audio(request: TranscriptionRequest) -> TranscriptionResponse:
    """
    Transcribe audio to text.
    
    This endpoint accepts a single audio file encoded in base64 and returns
    the transcribed text along with metadata about the processing.
    
    **Supported formats:** mp3, wav, flac, m4a, ogg
    **Maximum file size:** 100MB
    **Maximum duration:** 30 minutes
    """
    try:
        logger.info(f"Transcription request received for format: {request.format}")
        
        # Validate audio format
        if request.format not in config.supported_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format: {request.format}. "
                       f"Supported formats: {', '.join(config.supported_formats)}"
            )
        
        # Perform transcription
        response = await voxtral_service.transcribe_audio(request)
        
        logger.info(f"Transcription completed in {response.processing_time_ms}ms")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchTranscriptionResponse)
@limiter.limit("10/minute")  # 10 requests per minute for batch processing
async def batch_transcribe_audio(request: BatchTranscriptionRequest) -> BatchTranscriptionResponse:
    """
    Transcribe multiple audio files in a single request.
    
    This endpoint accepts multiple audio files and returns transcriptions
    for each file. Failed transcriptions will be marked as unsuccessful
    with error details.
    
    **Maximum files per request:** 10
    **Supported formats:** mp3, wav, flac, m4a, ogg
    **Maximum file size:** 100MB per file
    **Maximum duration:** 30 minutes per file
    """
    try:
        logger.info(f"Batch transcription request received for {len(request.audio_files)} files")
        
        # Validate number of files
        if len(request.audio_files) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 audio files allowed per batch request"
            )
        
        # Validate audio formats
        for i, audio_file in enumerate(request.audio_files):
            if audio_file.format not in config.supported_formats:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported audio format in file {i}: {audio_file.format}. "
                           f"Supported formats: {', '.join(config.supported_formats)}"
                )
        
        # Perform batch transcription
        response = await voxtral_service.batch_transcribe(request)
        
        successful_count = sum(1 for result in response.results if result.success)
        logger.info(f"Batch transcription completed: {successful_count}/{len(request.audio_files)} successful "
                   f"in {response.total_processing_time_ms}ms")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch transcription failed: {str(e)}"
        )
