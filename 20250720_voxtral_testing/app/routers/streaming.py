"""Streaming transcription endpoints."""

import asyncio
import base64
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from ..core.config import config
from ..core.logging import get_logger
from ..models.audio import AudioFormat, AudioLanguage
from ..services.voxtral_service import voxtral_service

logger = get_logger(__name__)
router = APIRouter(prefix="/stream", tags=["streaming"])


class StreamingTranscriptionRequest(BaseModel):
    """Request model for streaming transcription."""
    
    format: AudioFormat = Field(..., description="Audio chunk format")
    language: Optional[AudioLanguage] = Field(None, description="Audio language (auto-detected if not provided)")
    chunk_id: int = Field(..., description="Sequential chunk identifier")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    audio_data: str = Field(..., description="Base64-encoded audio data")
    
    @validator("audio_data")
    def validate_base64(cls, v: str) -> str:
        """Validate that the data is properly base64-encoded."""
        try:
            # Try to decode to check if it's valid base64
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64-encoded audio data")


class StreamingTranscriptionResponse(BaseModel):
    """Response model for streaming transcription."""
    
    chunk_id: int = Field(..., description="Chunk identifier this response corresponds to")
    transcription: str = Field(..., description="Transcribed text for this chunk")
    is_final: bool = Field(False, description="Whether this is a final transcription")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


@router.websocket("/transcribe")
async def websocket_transcribe(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for streaming audio transcription.
    
    This endpoint accepts audio chunks over a WebSocket connection and
    returns transcriptions in real-time. The client should send audio
    chunks as JSON objects with the following structure:
    
    ```json
    {
        "format": "mp3",
        "language": "en",
        "chunk_id": 1,
        "is_final": false,
        "audio_data": "base64_encoded_audio_data"
    }
    ```
    
    The server will respond with transcription results as they become
    available:
    
    ```json
    {
        "chunk_id": 1,
        "transcription": "Transcribed text for this chunk",
        "is_final": false,
        "confidence": 0.95,
        "processing_time_ms": 150
    }
    ```
    
    **Supported formats:** mp3, wav, flac, m4a, ogg
    **Recommended chunk size:** 1-3 seconds
    **Maximum chunk size:** 10MB
    """
    await websocket.accept()
    
    try:
        # Check if Voxtral backend is healthy
        is_healthy = await voxtral_service.health_check()
        
        if not is_healthy:
            await websocket.send_json({
                "error": {
                    "code": "MODEL_UNAVAILABLE",
                    "message": "Voxtral backend is not available",
                }
            })
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return
        
        # Process incoming audio chunks
        buffer: List[Dict[str, Any]] = []
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                # Parse message
                message = json.loads(data)
                request = StreamingTranscriptionRequest(**message)
                
                # Validate audio format
                if request.format not in config.supported_formats:
                    await websocket.send_json({
                        "error": {
                            "code": "INVALID_AUDIO_FORMAT",
                            "message": f"Unsupported audio format: {request.format}",
                        }
                    })
                    continue
                
                # Process audio chunk
                start_time = time.time()
                
                # For now, we'll use the regular transcription endpoint
                # In a production system, you would implement proper streaming
                # with VAD and chunk aggregation
                from ..models.audio import TranscriptionRequest as APITranscriptionRequest

                transcription_request = APITranscriptionRequest(
                    audio_file=request.audio_data,
                    format=request.format,
                    language=request.language,
                    temperature=0.0,
                )

                response = await voxtral_service.transcribe_audio(transcription_request)
                
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Send response
                await websocket.send_json({
                    "chunk_id": request.chunk_id,
                    "transcription": response.transcription,
                    "is_final": request.is_final,
                    "confidence": response.confidence,
                    "processing_time_ms": processing_time_ms,
                })
                
                # If this is the final chunk, close the connection
                if request.is_final:
                    await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                    break
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "error": {
                        "code": "INVALID_JSON",
                        "message": "Invalid JSON message",
                    }
                })
            
            except Exception as e:
                logger.error(f"Error processing streaming request: {e}")
                await websocket.send_json({
                    "error": {
                        "code": "PROCESSING_ERROR",
                        "message": str(e),
                    }
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass
