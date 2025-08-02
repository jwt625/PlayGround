"""Audio-related models for API requests and responses."""

import base64
from datetime import datetime
from typing import Dict, List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field, validator

# Type aliases
AudioFormat = Literal["mp3", "wav", "flac", "m4a", "ogg"]
AudioLanguage = Literal["en", "es", "fr", "pt", "hi", "de", "nl", "it"]


class AudioFile(BaseModel):
    """Audio file representation for API requests."""
    
    data: str = Field(..., description="Base64-encoded audio data")
    format: AudioFormat = Field(..., description="Audio file format")
    id: Optional[str] = Field(None, description="Optional client-provided identifier")
    
    @validator("data")
    def validate_base64(cls, v: str) -> str:
        """Validate that the data is properly base64-encoded."""
        try:
            # Try to decode to check if it's valid base64
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64-encoded audio data")


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    
    audio_file: str = Field(..., description="Base64-encoded audio data")
    format: AudioFormat = Field(..., description="Audio file format")
    language: Optional[AudioLanguage] = Field(None, description="Audio language (auto-detected if not provided)")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Temperature for generation (0.0 recommended for transcription)")


class TranscriptionResponse(BaseModel):
    """Response model for audio transcription."""
    
    transcription: str = Field(..., description="The transcribed text content")
    language_detected: Optional[str] = Field(None, description="Detected language code")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    audio_duration_seconds: Optional[float] = Field(None, description="Duration of the audio in seconds")


class AudioUnderstandingRequest(BaseModel):
    """Request model for audio understanding."""
    
    audio_files: List[AudioFile] = Field(..., min_items=1, description="List of audio files to analyze")
    question: str = Field(..., min_length=1, description="Question about the audio content")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(500, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")


class AudioUnderstandingResponse(BaseModel):
    """Response model for audio understanding."""
    
    answer: str = Field(..., description="Answer to the question about the audio")
    audio_count: int = Field(..., description="Number of audio files processed")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    token_usage: Dict[str, int] = Field(..., description="Token usage statistics")


class BatchTranscriptionRequest(BaseModel):
    """Request model for batch audio transcription."""
    
    audio_files: List[AudioFile] = Field(..., min_items=1, max_items=10, description="List of audio files to transcribe")
    language: Optional[AudioLanguage] = Field(None, description="Audio language (auto-detected if not provided)")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Temperature for generation")


class BatchTranscriptionResult(BaseModel):
    """Result for a single file in batch transcription."""
    
    id: str = Field(..., description="Client-provided identifier or auto-generated ID")
    transcription: Optional[str] = Field(None, description="The transcribed text content")
    error: Optional[str] = Field(None, description="Error message if transcription failed")
    success: bool = Field(..., description="Whether transcription was successful")


class BatchTranscriptionResponse(BaseModel):
    """Response model for batch audio transcription."""
    
    results: List[BatchTranscriptionResult] = Field(..., description="Results for each audio file")
    total_processing_time_ms: int = Field(..., description="Total processing time in milliseconds")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: Dict[str, Any] = Field(..., description="Error details")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    model: str = Field(..., description="Model name")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")
    version: str = Field(..., description="API version")
